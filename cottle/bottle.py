#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bottle is a fast and simple micro-framework for small web applications. It
offers request dispatching (Routes) with url parameter support, templates,
a built-in HTTP Server and adapters for many third party WSGI/HTTP-server and
template engines - all in a single file and with no dependencies other than the
Python Standard Library.

Homepage and documentation: http://bottlepy.org/

Copyright (c) 2014, Marcel Hellkamp.
License: MIT (see LICENSE for details)
"""

from __future__ import with_statement

__author__ = 'Marcel Hellkamp'
__version__ = '0.13-dev'
__license__ = 'MIT'

# The gevent and eventlet server adapters need to patch some modules before
# they are imported. This is why we parse the commandline parameters here but
# handle them later
if __name__ == '__main__':
    from optparse import OptionParser
    _cmd_parser = OptionParser(usage="usage: %prog [options] package.module:app")
    _opt = _cmd_parser.add_option
    _opt("--version", action="store_true", help="show version number.")
    _opt("-b", "--bind", metavar="ADDRESS", help="bind socket to ADDRESS.")
    _opt("-s", "--server", default='wsgiref', help="use SERVER as backend.")
    _opt("-p", "--plugin", action="append", help="install additional plugin/s.")
    _opt("--debug", action="store_true", help="start server in debug mode.")
    _opt("--reload", action="store_true", help="auto-reload on file changes.")
    _cmd_options, _cmd_args = _cmd_parser.parse_args()
    if _cmd_options.server:
        if _cmd_options.server.startswith('gevent'):
            import gevent.monkey; gevent.monkey.patch_all()
        elif _cmd_options.server.startswith('eventlet'):
            import eventlet; eventlet.monkey_patch()

import base64, cgi, email.utils, functools, hmac, imp, itertools, mimetypes,\
        os, re, subprocess, sys, tempfile, threading, time, warnings

from datetime import date as datedate, datetime, timedelta
from tempfile import TemporaryFile
from traceback import format_exc, print_exc
from inspect import getargspec
from unicodedata import normalize




from py23k import *
from py23k import _stderr#why
from comment import html_quote, html_escape, cached_property,DictProperty
from mapping import Mapping

from http_wsgi import BaseRequest
from http_wsgi import HeaderProperty 
from http_wsgi import _hkey
from http_wsgi import BaseResponse
from http_wsgi import HTTPError



# Some helpers for string/byte handling
def tob(s, enc='utf8'):
    return s.encode(enc) if isinstance(s, unicode) else bytes(s)


def touni(s, enc='utf8', err='strict'):
    if isinstance(s, bytes):
        return s.decode(enc, err)
    else:
        return unicode(s or ("" if s is None else s))

tonat = touni if py3k else tob

# 3.2 fixes cgi.FieldStorage to accept bytes (which makes a lot of sense).
# 3.1 needs a workaround.
if py31:
    from io import TextIOWrapper

    class NCTextIOWrapper(TextIOWrapper):
        def close(self): pass # Keep wrapped buffer open.


# A bug in functools causes it to break if the wrapper is an instance method
def update_wrapper(wrapper, wrapped, *a, **ka):
    try:
        functools.update_wrapper(wrapper, wrapped, *a, **ka)
    except AttributeError:
        pass


# These helpers are used at module level and need to be defined first.
# And yes, I know PEP-8, but sometimes a lower-case classname makes more sense.

def depr(message, strict=False):
    warnings.warn(message, DeprecationWarning, stacklevel=3)

def makelist(data): # This is just too handy
    if isinstance(data, (tuple, list, set, dict)):
        return list(data)
    elif data:
        return [data]
    else:
        return []






class lazy_attribute(object):
    """ A property that caches itself to the class object. """
    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value






###############################################################################
# Exceptions and Events ########################################################
###############################################################################


from Error import * 





###############################################################################
# Routing ######################################################################
###############################################################################




def _re_flatten(p):
    """ Turn all capturing groups in a regular expression pattern into
        non-capturing groups. """
    if '(' not in p:
        return p
    return re.sub(r'(\\*)(\(\?P<[^>]+>|\((?!\?))',
        lambda m: m.group(0) if len(m.group(1)) % 2 else m.group(1) + '(?:', p)


class Router(object):
    """ A Router is an ordered collection of route->target pairs. It is used to
        efficiently match WSGI requests against a number of routes and return
        the first target that satisfies the request. The target may be anything,
        usually a string, ID or callable object. A route consists of a path-rule
        and a HTTP method.

        The path-rule is either a static path (e.g. `/contact`) or a dynamic
        path that contains wildcards (e.g. `/wiki/<page>`). The wildcard syntax
        and details on the matching order are described in docs:`routing`.
    """

    default_pattern = '[^/]+'
    default_filter  = 're'

    #: The current CPython regexp implementation does not allow more
    #: than 99 matching groups per regular expression.
    _MAX_GROUPS_PER_PATTERN = 99

    def __init__(self, strict=False):
        self.rules    = [] # All rules in order
        self._groups  = {} # index of regexes to find them in dyna_routes
        self.builder  = {} # Data structure for the url builder
        self.static   = {} # Search structure for static routes
        self.dyna_routes   = {}
        self.dyna_regexes  = {} # Search structure for dynamic routes
        #: If true, static routes are no longer checked first.
        self.strict_order = strict
        self.filters = {
            're':    lambda conf:
                (_re_flatten(conf or self.default_pattern), None, None),
            'int':   lambda conf: (r'-?\d+', int, lambda x: str(int(x))),
            'float': lambda conf: (r'-?[\d.]+', float, lambda x: str(float(x))),
            'path':  lambda conf: (r'.+?', None, None)}

    def add_filter(self, name, func):
        """ Add a filter. The provided function is called with the configuration
        string as parameter and must return a (regexp, to_python, to_url) tuple.
        The first element is a string, the last two are callables or None. """
        self.filters[name] = func

    rule_syntax = re.compile('(\\\\*)'
        '(?:(?::([a-zA-Z_][a-zA-Z_0-9]*)?()(?:#(.*?)#)?)'
          '|(?:<([a-zA-Z_][a-zA-Z_0-9]*)?(?::([a-zA-Z_]*)'
            '(?::((?:\\\\.|[^\\\\>]+)+)?)?)?>))')

    def _itertokens(self, rule):
        offset, prefix = 0, ''
        for match in self.rule_syntax.finditer(rule):
            prefix += rule[offset:match.start()]
            g = match.groups()
            if len(g[0])%2: # Escaped wildcard
                prefix += match.group(0)[len(g[0]):]
                offset = match.end()
                continue
            if prefix:
                yield prefix, None, None
            name, filtr, conf = g[4:7] if g[2] is None else g[1:4]
            yield name, filtr or 'default', conf or None
            offset, prefix = match.end(), ''
        if offset <= len(rule) or prefix:
            yield prefix+rule[offset:], None, None

    def add(self, rule, method, target, name=None):
        """ Add a new rule or replace the target for an existing rule. """
        anons     = 0    # Number of anonymous wildcards found
        keys      = []   # Names of keys
        pattern   = ''   # Regular expression pattern with named groups
        filters   = []   # Lists of wildcard input filters
        builder   = []   # Data structure for the URL builder
        is_static = True

        for key, mode, conf in self._itertokens(rule):
            if mode:
                is_static = False
                if mode == 'default': mode = self.default_filter
                mask, in_filter, out_filter = self.filters[mode](conf)
                if not key:
                    pattern += '(?:%s)' % mask
                    key = 'anon%d' % anons
                    anons += 1
                else:
                    pattern += '(?P<%s>%s)' % (key, mask)
                    keys.append(key)
                if in_filter: filters.append((key, in_filter))
                builder.append((key, out_filter or str))
            elif key:
                pattern += re.escape(key)
                builder.append((None, key))

        self.builder[rule] = builder
        if name: self.builder[name] = builder

        if is_static and not self.strict_order:
            self.static.setdefault(method, {})
            self.static[method][self.build(rule)] = (target, None)
            return

        try:
            re_pattern = re.compile('^(%s)$' % pattern)
            re_match = re_pattern.match
        except re.error:
            raise RouteSyntaxError("Could not add Route: %s (%s)" % (rule, _e()))

        if filters:
            def getargs(path):
                url_args = re_match(path).groupdict()
                for name, wildcard_filter in filters:
                    try:
                        url_args[name] = wildcard_filter(url_args[name])
                    except ValueError:
                        raise HTTPError(400, 'Path has wrong format.')
                return url_args
        elif re_pattern.groupindex:
            def getargs(path):
                return re_match(path).groupdict()
        else:
            getargs = None

        flatpat = _re_flatten(pattern)
        whole_rule = (rule, flatpat, target, getargs)

        if (flatpat, method) in self._groups:
            if DEBUG:
                msg = 'Route <%s %s> overwrites a previously defined route'
                warnings.warn(msg % (method, rule), RuntimeWarning)
            self.dyna_routes[method][self._groups[flatpat, method]] = whole_rule
        else:
            self.dyna_routes.setdefault(method, []).append(whole_rule)
            self._groups[flatpat, method] = len(self.dyna_routes[method]) - 1

        self._compile(method)

    def _compile(self, method):
        all_rules = self.dyna_routes[method]
        comborules = self.dyna_regexes[method] = []
        maxgroups = self._MAX_GROUPS_PER_PATTERN
        for x in range(0, len(all_rules), maxgroups):
            some = all_rules[x:x+maxgroups]
            combined = (flatpat for (_, flatpat, _, _) in some)
            combined = '|'.join('(^%s$)' % flatpat for flatpat in combined)
            combined = re.compile(combined).match
            rules = [(target, getargs) for (_, _, target, getargs) in some]
            comborules.append((combined, rules))

    def build(self, _name, *anons, **query):
        """ Build an URL by filling the wildcards in a rule. """
        builder = self.builder.get(_name)
        if not builder: raise RouteBuildError("No route with that name.", _name)
        try:
            for i, value in enumerate(anons): query['anon%d'%i] = value
            url = ''.join([f(query.pop(n)) if n else f for (n,f) in builder])
            return url if not query else url+'?'+urlencode(query)
        except KeyError:
            raise RouteBuildError('Missing URL argument: %r' % _e().args[0])

    def match(self, environ):
        """ Return a (target, url_args) tuple or raise HTTPError(400/404/405). """
        verb = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'] or '/'

        if verb == 'HEAD':
            methods = ['PROXY', verb, 'GET', 'ANY']
        else:
            methods = ['PROXY', verb, 'ANY']

        for method in methods:
            if method in self.static and path in self.static[method]:
                target, getargs = self.static[method][path]
                return target, getargs(path) if getargs else {}
            elif method in self.dyna_regexes:
                for combined, rules in self.dyna_regexes[method]:
                    match = combined(path)
                    if match:
                        target, getargs = rules[match.lastindex - 1]
                        return target, getargs(path) if getargs else {}

        # No matching route found. Collect alternative methods for 405 response
        allowed = set([])
        nocheck = set(methods)
        for method in set(self.static) - nocheck:
            if path in self.static[method]:
                allowed.add(verb)
        for method in set(self.dyna_regexes) - allowed - nocheck:
            for combined, rules in self.dyna_regexes[method]:
                match = combined(path)
                if match:
                    allowed.add(method)
        if allowed:
            allow_header = ",".join(sorted(allowed))
            raise HTTPError(405, "Method not allowed.", Allow=allow_header)

        # No matching route and no alternative method found. We give up
        raise HTTPError(404, "Not found: " + repr(path))






class Route(object):
    """ This class wraps a route callback along with route specific metadata and
        configuration and applies Plugins on demand. It is also responsible for
        turing an URL path rule into a regular expression usable by the Router.
    """

    def __init__(self, app, rule, method, callback, name=None,
                 plugins=None, skiplist=None, **config):
        #: The application this route is installed to.
        self.app = app
        #: The path-rule string (e.g. ``/wiki/<page>``).
        self.rule = rule
        #: The HTTP method as a string (e.g. ``GET``).
        self.method = method
        #: The original callback with no plugins applied. Useful for introspection.
        self.callback = callback
        #: The name of the route (if specified) or ``None``.
        self.name = name or None
        #: A list of route-specific plugins (see :meth:`Bottle.route`).
        self.plugins = plugins or []
        #: A list of plugins to not apply to this route (see :meth:`Bottle.route`).
        self.skiplist = skiplist or []
        #: Additional keyword arguments passed to the :meth:`Bottle.route`
        #: decorator are stored in this dictionary. Used for route-specific
        #: plugin configuration and meta-data.
        self.config = ConfigDict().load_dict(config)

    @cached_property
    def call(self):
        """ The route callback with all plugins applied. This property is
            created on demand and then cached to speed up subsequent requests."""
        return self._make_callback()

    def reset(self):
        """ Forget any cached values. The next time :attr:`call` is accessed,
            all plugins are re-applied. """
        self.__dict__.pop('call', None)

    def prepare(self):
        """ Do all on-demand work immediately (useful for debugging)."""
        self.call

    def all_plugins(self):
        """ Yield all Plugins affecting this route. """
        unique = set()
        for p in reversed(self.app.plugins + self.plugins):
            if True in self.skiplist: break
            name = getattr(p, 'name', False)
            if name and (name in self.skiplist or name in unique): continue
            if p in self.skiplist or type(p) in self.skiplist: continue
            if name: unique.add(name)
            yield p

    def _make_callback(self):
        callback = self.callback
        for plugin in self.all_plugins():
            try:
                if hasattr(plugin, 'apply'):
                    callback = plugin.apply(callback, self)
                else:
                    callback = plugin(callback)
            except RouteReset: # Try again with changed configuration.
                return self._make_callback()
            if not callback is self.callback:
                update_wrapper(callback, self.callback)
        return callback

    def get_undecorated_callback(self):
        """ Return the callback. If the callback is a decorated function, try to
            recover the original function. """
        func = self.callback
        func = getattr(func, '__func__' if py3k else 'im_func', func)
        closure_attr = '__closure__' if py3k else 'func_closure'
        while hasattr(func, closure_attr) and getattr(func, closure_attr):
            func = getattr(func, closure_attr)[0].cell_contents
        return func

    def get_callback_args(self):
        """ Return a list of argument names the callback (most likely) accepts
            as keyword arguments. If the callback is a decorated function, try
            to recover the original function before inspection. """
        return getargspec(self.get_undecorated_callback())[0]

    def get_config(self, key, default=None):
        """ Lookup a config field and return its value, first checking the
            route.config, then route.app.config."""
        for conf in (self.config, self.app.conifg):
            if key in conf: return conf[key]
        return default

    def __repr__(self):
        cb = self.get_undecorated_callback()
        return '<%s %r %r>' % (self.method, self.rule, cb)









###############################################################################
# Application Object ###########################################################
###############################################################################


class Bottle(object):
    """ Each Bottle object represents a single, distinct web application and
        consists of routes, callbacks, plugins, resources and configuration.
        Instances are callable WSGI applications.

        :param catchall: If true (default), handle all exceptions. Turn off to
                         let debugging middleware handle exceptions.
    """

    def __init__(self, catchall=True, autojson=True):

        #: A :class:`ConfigDict` for app specific configuration.
        self.config = ConfigDict()
        self.config._on_change = functools.partial(self.trigger_hook, 'config')
        self.config.meta_set('autojson', 'validate', bool)
        self.config.meta_set('catchall', 'validate', bool)
        self.config['catchall'] = catchall
        self.config['autojson'] = autojson

        #: A :class:`ResourceManager` for application files
        self.resources = ResourceManager()

        self.routes = [] # List of installed :class:`Route` instances.
        self.router = Router() # Maps requests to :class:`Route` instances.
        self.error_handler = {}
        self.mapping = Mapping()  # mapping to handle call add by feng

        # Core plugins
        self.plugins = [] # List of installed plugins.
        if self.config['autojson']:
            self.install(JSONPlugin())
        self.install(TemplatePlugin())

    #: If true, most exceptions are caught and returned as :exc:`HTTPError`
    catchall = DictProperty('config', 'catchall')

    __hook_names = 'before_request', 'after_request', 'app_reset', 'config'
    __hook_reversed = 'after_request'

    @cached_property
    def _hooks(self):
        return dict((name, []) for name in self.__hook_names)

    def add_hook(self, name, func):
        """ Attach a callback to a hook. Three hooks are currently implemented:

            before_request
                Executed once before each request. The request context is
                available, but no routing has happened yet.
            after_request
                Executed once after each request regardless of its outcome.
            app_reset
                Called whenever :meth:`Bottle.reset` is called.
        """
        if name in self.__hook_reversed:
            self._hooks[name].insert(0, func)
        else:
            self._hooks[name].append(func)

    def remove_hook(self, name, func):
        """ Remove a callback from a hook. """
        if name in self._hooks and func in self._hooks[name]:
            self._hooks[name].remove(func)
            return True

    def trigger_hook(self, __name, *args, **kwargs):
        """ Trigger a hook and return a list of results. """
        return [hook(*args, **kwargs) for hook in self._hooks[__name][:]]

    def hook(self, name):
        """ Return a decorator that attaches a callback to a hook. See
            :meth:`add_hook` for details."""
        def decorator(func):
            self.add_hook(name, func)
            return func
        return decorator

    def mount(self, prefix, app, **options):
        """ Mount an application (:class:`Bottle` or plain WSGI) to a specific
            URL prefix. Example::

                root_app.mount('/admin/', admin_app)

            :param prefix: path prefix or `mount-point`. If it ends in a slash,
                that slash is mandatory.
            :param app: an instance of :class:`Bottle` or a WSGI application.

            All other parameters are passed to the underlying :meth:`route` call.
        """

        segments = [p for p in prefix.split('/') if p]
        if not segments: raise ValueError('Empty path prefix.')
        path_depth = len(segments)

        def mountpoint_wrapper():
            try:
                request.path_shift(path_depth)
                rs = HTTPResponse([])
                def start_response(status, headerlist, exc_info=None):
                    if exc_info:
                        _raise(*exc_info)
                    rs.status = status
                    for name, value in headerlist: rs.add_header(name, value)
                    return rs.body.append
                body = app(request.environ, start_response)
                if body and rs.body: body = itertools.chain(rs.body, body)
                rs.body = body or rs.body
                return rs
            finally:
                request.path_shift(-path_depth)

        options.setdefault('skip', True)
        options.setdefault('method', 'PROXY')
        options.setdefault('mountpoint', {'prefix': prefix, 'target': app})
        options['callback'] = mountpoint_wrapper

        self.route('/%s/<:re:.*>' % '/'.join(segments), **options)
        if not prefix.endswith('/'):
            self.route('/' + '/'.join(segments), **options)

    def merge(self, routes):
        """ Merge the routes of another :class:`Bottle` application or a list of
            :class:`Route` objects into this application. The routes keep their
            'owner', meaning that the :data:`Route.app` attribute is not
            changed. """
        if isinstance(routes, Bottle):
            routes = routes.routes
        for route in routes:
            self.add_route(route)

    def install(self, plugin):
        """ Add a plugin to the list of plugins and prepare it for being
            applied to all routes of this application. A plugin may be a simple
            decorator or an object that implements the :class:`Plugin` API.
        """
        if hasattr(plugin, 'setup'): plugin.setup(self)
        if not callable(plugin) and not hasattr(plugin, 'apply'):
            raise TypeError("Plugins must be callable or implement .apply()")
        self.plugins.append(plugin)
        self.reset()
        return plugin

    def uninstall(self, plugin):
        """ Uninstall plugins. Pass an instance to remove a specific plugin, a type
            object to remove all plugins that match that type, a string to remove
            all plugins with a matching ``name`` attribute or ``True`` to remove all
            plugins. Return the list of removed plugins. """
        removed, remove = [], plugin
        for i, plugin in list(enumerate(self.plugins))[::-1]:
            if remove is True or remove is plugin or remove is type(plugin) \
            or getattr(plugin, 'name', True) == remove:
                removed.append(plugin)
                del self.plugins[i]
                if hasattr(plugin, 'close'): plugin.close()
        if removed: self.reset()
        return removed

    def reset(self, route=None):
        """ Reset all routes (force plugins to be re-applied) and clear all
            caches. If an ID or route object is given, only that specific route
            is affected. """
        if route is None: routes = self.routes
        elif isinstance(route, Route): routes = [route]
        else: routes = [self.routes[route]]
        for route in routes: route.reset()
        if DEBUG:
            for route in routes: route.prepare()
        self.trigger_hook('app_reset')

    def close(self):
        """ Close the application and all installed plugins. """
        for plugin in self.plugins:
            if hasattr(plugin, 'close'): plugin.close()

    def run(self, **kwargs):
        """ Calls :func:`run` with the same parameters. """
        run(self, **kwargs)

    def match(self, environ):
        """ Search for a matching route and return a (:class:`Route` , urlargs)
            tuple. The second value is a dictionary with parameters extracted
            from the URL. Raise :exc:`HTTPError` (404/405) on a non-match."""
        return self.router.match(environ)

    def get_url(self, routename, **kargs):
        """ Return a string that matches a named route """
        scriptname = request.environ.get('SCRIPT_NAME', '').strip('/') + '/'
        location = self.router.build(routename, **kargs).lstrip('/')
        return urljoin(urljoin('/', scriptname), location)

    def add_route(self, route):
        """ Add a route object, but do not change the :data:`Route.app`
            attribute."""
        self.routes.append(route)
        self.router.add(route.rule, route.method, route, name=route.name)
        if DEBUG: route.prepare()

    def route(self, path=None, method='GET', callback=None, name=None,
              apply=None, skip=None, **config):
        """ A decorator to bind a function to a request URL. Example::

                @app.route('/hello/<name>')
                def hello(name):
                    return 'Hello %s' % name

            The ``:name`` part is a wildcard. See :class:`Router` for syntax
            details.

            :param path: Request path or a list of paths to listen to. If no
              path is specified, it is automatically generated from the
              signature of the function.
            :param method: HTTP method (`GET`, `POST`, `PUT`, ...) or a list of
              methods to listen to. (default: `GET`)
            :param callback: An optional shortcut to avoid the decorator
              syntax. ``route(..., callback=func)`` equals ``route(...)(func)``
            :param name: The name for this route. (default: None)
            :param apply: A decorator or plugin or a list of plugins. These are
              applied to the route callback in addition to installed plugins.
            :param skip: A list of plugins, plugin classes or names. Matching
              plugins are not installed to this route. ``True`` skips all.

            Any additional keyword arguments are stored as route-specific
            configuration and passed to plugins (see :meth:`Plugin.apply`).
        """
        if callable(path): path, callback = None, path
        plugins = makelist(apply)
        skiplist = makelist(skip)
        def decorator(callback):
            if isinstance(callback, basestring): callback = load(callback)
            for rule in makelist(path) or yieldroutes(callback):
                for verb in makelist(method):
                    verb = verb.upper()
                    route = Route(self, rule, verb, callback, name=name,
                                  plugins=plugins, skiplist=skiplist, **config)
                    self.add_route(route)
            return callback
        return decorator(callback) if callback else decorator

    def get(self, path=None, method='GET', **options):
        """ Equals :meth:`route`. """
        return self.route(path, method, **options)

    def post(self, path=None, method='POST', **options):
        """ Equals :meth:`route` with a ``POST`` method parameter. """
        return self.route(path, method, **options)

    def put(self, path=None, method='PUT', **options):
        """ Equals :meth:`route` with a ``PUT`` method parameter. """
        return self.route(path, method, **options)

    def delete(self, path=None, method='DELETE', **options):
        """ Equals :meth:`route` with a ``DELETE`` method parameter. """
        return self.route(path, method, **options)

    def patch(self, path=None, method='PATCH', **options):
        """ Equals :meth:`route` with a ``PATCH`` method parameter. """
        return self.route(path, method, **options)

    def error(self, code=500):
        """ Decorator: Register an output handler for a HTTP error code"""
        def wrapper(handler):
            self.error_handler[int(code)] = handler
            return handler
        return wrapper

    def default_error_handler(self, res):
        return tob(template(ERROR_PAGE_TEMPLATE, e=res))

    def _handle(self, environ):
        path = environ['bottle.raw_path'] = environ['PATH_INFO']
        if py3k:
            try:
                environ['PATH_INFO'] = path.encode('latin1').decode('utf8')
            except UnicodeError:
                return HTTPError(400, 'Invalid path string. Expected UTF-8')

        try:
            environ['bottle.app'] = self
            request.bind(environ)
            response.bind()
            try:
                self.trigger_hook('before_request')

                handle, args = self.mapping.match(path)
                if handle:
                    environ['route.handle'] = handle
                    environ['bottle.route'] = handle
                    environ['route.url_args'] = args
                    res  = self.mapping.call(handle, args, request)
                    return  res
                else:
                    route, args = self.router.match(environ)
                    environ['route.handle'] = route
                    environ['bottle.route'] = route
                    environ['route.url_args'] = args
                    return route.call(**args)
            finally:
                self.trigger_hook('after_request')
        except HTTPResponse:
            return _e()
        except RouteReset:
            route.reset()
            return self._handle(environ)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception:
            if not self.catchall: raise
            stacktrace = format_exc()
            environ['wsgi.errors'].write(stacktrace)
            return HTTPError(500, "Internal Server Error", _e(), stacktrace)

    def _cast(self, out, peek=None):
        """ Try to convert the parameter into something WSGI compatible and set
        correct HTTP headers when possible.
        Support: False, str, unicode, dict, HTTPResponse, HTTPError, file-like,
        iterable of strings and iterable of unicodes
        """

        # Empty output is done here
        if not out:
            if 'Content-Length' not in response:
                response['Content-Length'] = 0
            return []
        # Join lists of byte or unicode strings. Mixed lists are NOT supported
        if isinstance(out, (tuple, list))\
        and isinstance(out[0], (bytes, unicode)):
            out = out[0][0:0].join(out) # b'abc'[0:0] -> b''
        # Encode unicode strings
        if isinstance(out, unicode):
            out = out.encode(response.charset)
        # Byte Strings are just returned
        if isinstance(out, bytes):
            if 'Content-Length' not in response:
                response['Content-Length'] = len(out)
            return [out]
        # HTTPError or HTTPException (recursive, because they may wrap anything)
        # TODO: Handle these explicitly in handle() or make them iterable.
        if isinstance(out, HTTPError):
            out.apply(response)
            out = self.error_handler.get(out.status_code, self.default_error_handler)(out)
            return self._cast(out)
        if isinstance(out, HTTPResponse):
            out.apply(response)
            return self._cast(out.body)

        # File-like objects.
        if hasattr(out, 'read'):
            if 'wsgi.file_wrapper' in request.environ:
                return request.environ['wsgi.file_wrapper'](out)
            elif hasattr(out, 'close') or not hasattr(out, '__iter__'):
                return WSGIFileWrapper(out)

        # Handle Iterables. We peek into them to detect their inner type.
        try:
            iout = iter(out)
            first = next(iout)
            while not first:
                first = next(iout)
        except StopIteration:
            return self._cast('')
        except HTTPResponse:
            first = _e()
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except:
            if not self.catchall: raise
            first = HTTPError(500, 'Unhandled exception', _e(), format_exc())

        # These are the inner types allowed in iterator or generator objects.
        if isinstance(first, HTTPResponse):
            return self._cast(first)
        elif isinstance(first, bytes):
            new_iter = itertools.chain([first], iout)
        elif isinstance(first, unicode):
            encoder = lambda x: x.encode(response.charset)
            new_iter = imap(encoder, itertools.chain([first], iout))
        else:
            msg = 'Unsupported response type: %s' % type(first)
            return self._cast(HTTPError(500, msg))
        if hasattr(out, 'close'):
            new_iter = _closeiter(new_iter, out.close)
        return new_iter

    def wsgi(self, environ, start_response):
        """ The bottle WSGI-interface. """
        try:
            out = self._cast(self._handle(environ))
            # rfc2616 section 4.3
            if response._status_code in (100, 101, 204, 304)\
            or environ['REQUEST_METHOD'] == 'HEAD':
                if hasattr(out, 'close'): out.close()
                out = []
            start_response(response._status_line, response.headerlist)
            return out
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except:
            if not self.catchall: raise
            err = '<h1>Critical error while processing request: %s</h1>' \
                  % html_escape(environ.get('PATH_INFO', '/'))
            if DEBUG:
                err += '<h2>Error:</h2>\n<pre>\n%s\n</pre>\n' \
                       '<h2>Traceback:</h2>\n<pre>\n%s\n</pre>\n' \
                       % (html_escape(repr(_e())), html_escape(format_exc()))
            environ['wsgi.errors'].write(err)
            headers = [('Content-Type', 'text/html; charset=UTF-8')]
            start_response('500 INTERNAL SERVER ERROR', headers, sys.exc_info())
            return [tob(err)]

    def __call__(self, environ, start_response):
        """ Each instance of :class:'Bottle' is a WSGI application. """
        return self.wsgi(environ, start_response)

    def __enter__(self):
        """ Use this application as default for all module-level shortcuts. """
        default_app.push(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        default_app.pop()





###############################################################################
# HTTP and WSGI Tools ##########################################################
###############################################################################


def _local_property():
    ls = threading.local()
    def fget(_):
        try: return ls.var
        except AttributeError:
            raise RuntimeError("Request context not initialized.")
    def fset(_, value): ls.var = value
    def fdel(_): del ls.var
    return property(fget, fset, fdel, 'Thread-local property')


class LocalRequest(BaseRequest):
    """ A thread-local subclass of :class:`BaseRequest` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`request`). If accessed during a
        request/response cycle, this instance always refers to the *current*
        request (even on a multithreaded server). """
    bind = BaseRequest.__init__
    environ = _local_property()


class LocalResponse(BaseResponse):
    """ A thread-local subclass of :class:`BaseResponse` with a different
        set of attributes for each thread. There is usually only one global
        instance of this class (:data:`response`). Its attributes are used
        to build the HTTP response at the end of the request/response cycle.
    """
    bind = BaseResponse.__init__
    _status_line = _local_property()
    _status_code = _local_property()
    _cookies     = _local_property()
    _headers     = _local_property()
    body         = _local_property()


#Request = BaseRequest
#Response = BaseResponse








###############################################################################
# Plugins ######################################################################
###############################################################################

class PluginError(BottleException): pass


class JSONPlugin(object):
    name = 'json'
    api  = 2

    def __init__(self, json_dumps=json_dumps):
        self.json_dumps = json_dumps

    def apply(self, callback, _):
        dumps = self.json_dumps
        if not dumps: return callback
        def wrapper(*a, **ka):
            try:
                rv = callback(*a, **ka)
            except HTTPError:
                rv = _e()

            if isinstance(rv, dict):
                #Attempt to serialize, raises exception on failure
                json_response = dumps(rv)
                #Set content type only if serialization successful
                response.content_type = 'application/json'
                return json_response
            elif isinstance(rv, HTTPResponse) and isinstance(rv.body, dict):
                rv.body = dumps(rv.body)
                rv.content_type = 'application/json'
            return rv

        return wrapper


class TemplatePlugin(object):
    """ This plugin applies the :func:`view` decorator to all routes with a
        `template` config parameter. If the parameter is a tuple, the second
        element must be a dict with additional options (e.g. `template_engine`)
        or default variables for the template. """
    name = 'template'
    api  = 2

    def apply(self, callback, route):
        conf = route.config.get('template')
        if isinstance(conf, (tuple, list)) and len(conf) == 2:
            return view(conf[0], **conf[1])(callback)
        elif isinstance(conf, str):
            return view(conf)(callback)
        else:
            return callback


#: Not a plugin, but part of the plugin API. TODO: Find a better place.
class _ImportRedirect(object):
    def __init__(self, name, impmask):
        """ Create a virtual package that redirects imports (see PEP 302). """
        self.name = name
        self.impmask = impmask
        self.module = sys.modules.setdefault(name, imp.new_module(name))
        self.module.__dict__.update({'__file__': __file__, '__path__': [],
                                    '__all__': [], '__loader__': self})
        sys.meta_path.append(self)

    def find_module(self, fullname, path=None):
        if '.' not in fullname: return
        packname = fullname.rsplit('.', 1)[0]
        if packname != self.name: return
        return self

    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        modname = fullname.rsplit('.', 1)[1]
        realname = self.impmask % modname
        __import__(realname)
        module = sys.modules[fullname] = sys.modules[realname]
        setattr(self.module, modname, module)
        module.__loader__ = self
        return module






###############################################################################
# Common Utilities #############################################################
###############################################################################


class MultiDict(DictMixin):
    """ This dict stores multiple values per key, but behaves exactly like a
        normal dict in that it returns only the newest value for any given key.
        There are special methods available to access the full list of values.
    """

    def __init__(self, *a, **k):
        self.dict = dict((k, [v]) for (k, v) in dict(*a, **k).items())

    def __len__(self): return len(self.dict)
    def __iter__(self): return iter(self.dict)
    def __contains__(self, key): return key in self.dict
    def __delitem__(self, key): del self.dict[key]
    def __getitem__(self, key): return self.dict[key][-1]
    def __setitem__(self, key, value): self.append(key, value)
    def keys(self): return self.dict.keys()

    if py3k:
        def values(self): return (v[-1] for v in self.dict.values())
        def items(self): return ((k, v[-1]) for k, v in self.dict.items())
        def allitems(self):
            return ((k, v) for k, vl in self.dict.items() for v in vl)
        iterkeys = keys
        itervalues = values
        iteritems = items
        iterallitems = allitems

    else:
        def values(self): return [v[-1] for v in self.dict.values()]
        def items(self): return [(k, v[-1]) for k, v in self.dict.items()]
        def iterkeys(self): return self.dict.iterkeys()
        def itervalues(self): return (v[-1] for v in self.dict.itervalues())
        def iteritems(self):
            return ((k, v[-1]) for k, v in self.dict.iteritems())
        def iterallitems(self):
            return ((k, v) for k, vl in self.dict.iteritems() for v in vl)
        def allitems(self):
            return [(k, v) for k, vl in self.dict.iteritems() for v in vl]

    def get(self, key, default=None, index=-1, type=None):
        """ Return the most recent value for a key.

            :param default: The default value to be returned if the key is not
                   present or the type conversion fails.
            :param index: An index for the list of available values.
            :param type: If defined, this callable is used to cast the value
                    into a specific type. Exception are suppressed and result in
                    the default value to be returned.
        """
        try:
            val = self.dict[key][index]
            return type(val) if type else val
        except Exception:
            pass
        return default

    def append(self, key, value):
        """ Add a new value to the list of values for this key. """
        self.dict.setdefault(key, []).append(value)

    def replace(self, key, value):
        """ Replace the list of values with a single value. """
        self.dict[key] = [value]

    def getall(self, key):
        """ Return a (possibly empty) list of values for a key. """
        return self.dict.get(key) or []

    #: Aliases for WTForms to mimic other multi-dict APIs (Django)
    getone = get
    getlist = getall


class FormsDict(MultiDict):
    """ This :class:`MultiDict` subclass is used to store request form data.
        Additionally to the normal dict-like item access methods (which return
        unmodified data as native strings), this container also supports
        attribute-like access to its values. Attributes are automatically de-
        or recoded to match :attr:`input_encoding` (default: 'utf8'). Missing
        attributes default to an empty string. """

    #: Encoding used for attribute values.
    input_encoding = 'utf8'
    #: If true (default), unicode strings are first encoded with `latin1`
    #: and then decoded to match :attr:`input_encoding`.
    recode_unicode = True

    def _fix(self, s, encoding=None):
        if isinstance(s, unicode) and self.recode_unicode: # Python 3 WSGI
            return s.encode('latin1').decode(encoding or self.input_encoding)
        elif isinstance(s, bytes): # Python 2 WSGI
            return s.decode(encoding or self.input_encoding)
        else:
            return s

    def decode(self, encoding=None):
        """ Returns a copy with all keys and values de- or recoded to match
            :attr:`input_encoding`. Some libraries (e.g. WTForms) want a
            unicode dictionary. """
        copy = FormsDict()
        enc = copy.input_encoding = encoding or self.input_encoding
        copy.recode_unicode = False
        for key, value in self.allitems():
            copy.append(self._fix(key, enc), self._fix(value, enc))
        return copy

    def getunicode(self, name, default=None, encoding=None):
        """ Return the value as a unicode string, or the default. """
        try:
            return self._fix(self[name], encoding)
        except (UnicodeError, KeyError):
            return default

    def __getattr__(self, name, default=unicode()):
        # Without this guard, pickle generates a cryptic TypeError:
        if name.startswith('__') and name.endswith('__'):
            return super(FormsDict, self).__getattr__(name)
        return self.getunicode(name, default=default)


class HeaderDict(MultiDict):
    """ A case-insensitive version of :class:`MultiDict` that defaults to
        replace the old value instead of appending it. """

    def __init__(self, *a, **ka):
        self.dict = {}
        if a or ka: self.update(*a, **ka)

    def __contains__(self, key): return _hkey(key) in self.dict
    def __delitem__(self, key): del self.dict[_hkey(key)]
    def __getitem__(self, key): return self.dict[_hkey(key)][-1]
    def __setitem__(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def append(self, key, value):
        self.dict.setdefault(_hkey(key), []).append(str(value))
    def replace(self, key, value): self.dict[_hkey(key)] = [str(value)]
    def getall(self, key): return self.dict.get(_hkey(key)) or []
    def get(self, key, default=None, index=-1):
        return MultiDict.get(self, _hkey(key), default, index)
    def filter(self, names):
        for name in [_hkey(n) for n in names]:
            if name in self.dict:
                del self.dict[name]


class WSGIHeaderDict(DictMixin):
    """ This dict-like class wraps a WSGI environ dict and provides convenient
        access to HTTP_* fields. Keys and values are native strings
        (2.x bytes or 3.x unicode) and keys are case-insensitive. If the WSGI
        environment contains non-native string values, these are de- or encoded
        using a lossless 'latin1' character set.

        The API will remain stable even on changes to the relevant PEPs.
        Currently PEP 333, 444 and 3333 are supported. (PEP 444 is the only one
        that uses non-native strings.)
    """
    #: List of keys that do not have a ``HTTP_`` prefix.
    cgikeys = ('CONTENT_TYPE', 'CONTENT_LENGTH')

    def __init__(self, environ):
        self.environ = environ

    def _ekey(self, key):
        """ Translate header field name to CGI/WSGI environ key. """
        key = key.replace('-','_').upper()
        if key in self.cgikeys:
            return key
        return 'HTTP_' + key

    def raw(self, key, default=None):
        """ Return the header value as is (may be bytes or unicode). """
        return self.environ.get(self._ekey(key), default)

    def __getitem__(self, key):
        val = self.environ[self._ekey(key)]
        if py3k:
            if isinstance(val, unicode):
                val = val.encode('latin1').decode('utf8')
            else:
                val = val.decode('utf8')
        return val

    def __setitem__(self, key, value):
        raise TypeError("%s is read-only." % self.__class__)

    def __delitem__(self, key):
        raise TypeError("%s is read-only." % self.__class__)

    def __iter__(self):
        for key in self.environ:
            if key[:5] == 'HTTP_':
                yield _hkey(key[5:])
            elif key in self.cgikeys:
                yield _hkey(key)

    def keys(self): return [x for x in self]
    def __len__(self): return len(self.keys())
    def __contains__(self, key): return self._ekey(key) in self.environ



class ConfigDict(dict):
    """ A dict-like configuration storage with additional support for
        namespaces, validators, meta-data, on_change listeners and more.
    """

    __slots__ = ('_meta', '_on_change')

    def __init__(self):
        self._meta = {}
        self._on_change = lambda name, value: None

    def load_config(self, filename):
        """ Load values from an ``*.ini`` style config file.

            If the config file contains sections, their names are used as
            namespaces for the values within. The two special sections
            ``DEFAULT`` and ``bottle`` refer to the root namespace (no prefix).
        """
        conf = ConfigParser()
        conf.read(filename)
        for section in conf.sections():
            for key, value in conf.items(section):
                if section not in ('DEFAULT', 'bottle'):
                    key = section + '.' + key
                self[key] = value
        return self

    def load_dict(self, source, namespace=''):
        """ Load values from a dictionary structure. Nesting can be used to
            represent namespaces.

            >>> c = ConfigDict()
            >>> c.load_dict({'some': {'namespace': {'key': 'value'} } })
            {'some.namespace.key': 'value'}
        """
        for key, value in source.items():
            if isinstance(key, str):
                nskey = (namespace + '.' + key).strip('.')
                if isinstance(value, dict):
                    self.load_dict(value, namespace=nskey)
                else:
                    self[nskey] = value
            else:
                raise TypeError('Key has type %r (not a string)' % type(key))
        return self

    def update(self, *a, **ka):
        """ If the first parameter is a string, all keys are prefixed with this
            namespace. Apart from that it works just as the usual dict.update().
            Example: ``update('some.namespace', key='value')`` """
        prefix = ''
        if a and isinstance(a[0], str):
            prefix = a[0].strip('.') + '.'
            a = a[1:]
        for key, value in dict(*a, **ka).items():
            self[prefix+key] = value

    def setdefault(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key has type %r (not a string)' % type(key))
        value = self.meta_get(key, 'filter', lambda x: x)(value)
        if key in self and self[key] is value:
            return
        self._on_change(key, value)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._on_change(key, None)
        dict.__delitem__(self, key)

    def meta_get(self, key, metafield, default=None):
        """ Return the value of a meta field for a key. """
        return self._meta.get(key, {}).get(metafield, default)

    def meta_set(self, key, metafield, value):
        """ Set the meta field for a key to a new value. This triggers the
            on-change handler for existing keys. """
        self._meta.setdefault(key, {})[metafield] = value
        if key in self:
            self[key] = self[key]

    def meta_list(self, key):
        """ Return an iterable of meta field names defined for a key. """
        return self._meta.get(key, {}).keys()


class AppStack(list):
    """ A stack-like list. Calling it returns the head of the stack. """

    def __call__(self):
        """ Return the current default application. """
        return self[-1]

    def push(self, value=None):
        """ Add a new :class:`Bottle` instance to the stack """
        if not isinstance(value, Bottle):
            value = Bottle()
        self.append(value)
        return value


class WSGIFileWrapper(object):

    def __init__(self, fp, buffer_size=1024*64):
        self.fp, self.buffer_size = fp, buffer_size
        for attr in ('fileno', 'close', 'read', 'readlines', 'tell', 'seek'):
            if hasattr(fp, attr): setattr(self, attr, getattr(fp, attr))

    def __iter__(self):
        buff, read = self.buffer_size, self.read
        while True:
            part = read(buff)
            if not part: return
            yield part


class _closeiter(object):
    """ This only exists to be able to attach a .close method to iterators that
        do not support attribute assignment (most of itertools). """

    def __init__(self, iterator, close=None):
        self.iterator = iterator
        self.close_callbacks = makelist(close)

    def __iter__(self):
        return iter(self.iterator)

    def close(self):
        for func in self.close_callbacks:
            func()


class ResourceManager(object):
    """ This class manages a list of search paths and helps to find and open
        application-bound resources (files).

        :param base: default value for :meth:`add_path` calls.
        :param opener: callable used to open resources.
        :param cachemode: controls which lookups are cached. One of 'all',
                         'found' or 'none'.
    """

    def __init__(self, base='./', opener=open, cachemode='all'):
        self.opener = opener
        self.base = base
        self.cachemode = cachemode

        #: A list of search paths. See :meth:`add_path` for details.
        self.path = []
        #: A cache for resolved paths. ``res.cache.clear()`` clears the cache.
        self.cache = {}

    def add_path(self, path, base=None, index=None, create=False):
        """ Add a new path to the list of search paths. Return False if the
            path does not exist.

            :param path: The new search path. Relative paths are turned into
                an absolute and normalized form. If the path looks like a file
                (not ending in `/`), the filename is stripped off.
            :param base: Path used to absolutize relative search paths.
                Defaults to :attr:`base` which defaults to ``os.getcwd()``.
            :param index: Position within the list of search paths. Defaults
                to last index (appends to the list).

            The `base` parameter makes it easy to reference files installed
            along with a python module or package::

                res.add_path('./resources/', __file__)
        """
        base = os.path.abspath(os.path.dirname(base or self.base))
        path = os.path.abspath(os.path.join(base, os.path.dirname(path)))
        path += os.sep
        if path in self.path:
            self.path.remove(path)
        if create and not os.path.isdir(path):
            os.makedirs(path)
        if index is None:
            self.path.append(path)
        else:
            self.path.insert(index, path)
        self.cache.clear()
        return os.path.exists(path)

    def __iter__(self):
        """ Iterate over all existing files in all registered paths. """
        search = self.path[:]
        while search:
            path = search.pop()
            if not os.path.isdir(path): continue
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isdir(full): search.append(full)
                else: yield full

    def lookup(self, name):
        """ Search for a resource and return an absolute file path, or `None`.

            The :attr:`path` list is searched in order. The first match is
            returend. Symlinks are followed. The result is cached to speed up
            future lookups. """
        if name not in self.cache or DEBUG:
            for path in self.path:
                fpath = os.path.join(path, name)
                if os.path.isfile(fpath):
                    if self.cachemode in ('all', 'found'):
                        self.cache[name] = fpath
                    return fpath
            if self.cachemode == 'all':
                self.cache[name] = None
        return self.cache[name]

    def open(self, name, mode='r', *args, **kwargs):
        """ Find a resource and return a file object, or raise IOError. """
        fname = self.lookup(name)
        if not fname: raise IOError("Resource %r not found." % name)
        return self.opener(fname, mode=mode, *args, **kwargs)


class FileUpload(object):

    def __init__(self, fileobj, name, filename, headers=None):
        """ Wrapper for file uploads. """
        #: Open file(-like) object (BytesIO buffer or temporary file)
        self.file = fileobj
        #: Name of the upload form field
        self.name = name
        #: Raw filename as sent by the client (may contain unsafe characters)
        self.raw_filename = filename
        #: A :class:`HeaderDict` with additional headers (e.g. content-type)
        self.headers = HeaderDict(headers) if headers else HeaderDict()

    content_type = HeaderProperty('Content-Type')
    content_length = HeaderProperty('Content-Length', reader=int, default=-1)

    @cached_property
    def filename(self):
        """ Name of the file on the client file system, but normalized to ensure
            file system compatibility. An empty filename is returned as 'empty'.

            Only ASCII letters, digits, dashes, underscores and dots are
            allowed in the final filename. Accents are removed, if possible.
            Whitespace is replaced by a single dash. Leading or tailing dots
            or dashes are removed. The filename is limited to 255 characters.
        """
        fname = self.raw_filename
        if not isinstance(fname, unicode):
            fname = fname.decode('utf8', 'ignore')
        fname = normalize('NFKD', fname).encode('ASCII', 'ignore').decode('ASCII')
        fname = os.path.basename(fname.replace('\\', os.path.sep))
        fname = re.sub(r'[^a-zA-Z0-9-_.\s]', '', fname).strip()
        fname = re.sub(r'[-\s]+', '-', fname).strip('.-')
        return fname[:255] or 'empty'

    def _copy_file(self, fp, chunk_size=2**16):
        read, write, offset = self.file.read, fp.write, self.file.tell()
        while 1:
            buf = read(chunk_size)
            if not buf: break
            write(buf)
        self.file.seek(offset)

    def save(self, destination, overwrite=False, chunk_size=2**16):
        """ Save file to disk or copy its content to an open file(-like) object.
            If *destination* is a directory, :attr:`filename` is added to the
            path. Existing files are not overwritten by default (IOError).

            :param destination: File path, directory or file(-like) object.
            :param overwrite: If True, replace existing files. (default: False)
            :param chunk_size: Bytes to read at a time. (default: 64kb)
        """
        if isinstance(destination, basestring): # Except file-likes here
            if os.path.isdir(destination):
                destination = os.path.join(destination, self.filename)
            if not overwrite and os.path.exists(destination):
                raise IOError('File exists.')
            with open(destination, 'wb') as fp:
                self._copy_file(fp, chunk_size)
        else:
            self._copy_file(destination, chunk_size)






###############################################################################
# Application Helper ###########################################################
###############################################################################


def abort(code=500, text='Unknown Error.'):
    """ Aborts execution and causes a HTTP error. """
    raise HTTPError(code, text)


def redirect(url, code=None):
    """ Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. """
    if not code:
        code = 303 if request.get('SERVER_PROTOCOL') == "HTTP/1.1" else 302
    res = response.copy(cls=HTTPResponse)
    res.status = code
    res.body = ""
    res.set_header('Location', urljoin(request.url, url))
    raise res


def _file_iter_range(fp, offset, bytes, maxread=1024*1024):
    """ Yield chunks from a range in a file. No chunk is bigger than maxread."""
    fp.seek(offset)
    while bytes > 0:
        part = fp.read(min(bytes, maxread))
        if not part: break
        bytes -= len(part)
        yield part


def static_file(filename, root, mimetype='auto', download=False, charset='UTF-8'):
    """ Open a file in a safe way and return :exc:`HTTPResponse` with status
        code 200, 305, 403 or 404. The ``Content-Type``, ``Content-Encoding``,
        ``Content-Length`` and ``Last-Modified`` headers are set if possible.
        Special support for ``If-Modified-Since``, ``Range`` and ``HEAD``
        requests.

        :param filename: Name or path of the file to send.
        :param root: Root path for file lookups. Should be an absolute directory
            path.
        :param mimetype: Defines the content-type header (default: guess from
            file extension)
        :param download: If True, ask the browser to open a `Save as...` dialog
            instead of opening the file with the associated program. You can
            specify a custom filename as a string. If not specified, the
            original filename is used (default: False).
        :param charset: The charset to use for files with a ``text/*``
            mime-type. (default: UTF-8)
    """

    root = os.path.abspath(root) + os.sep
    filename = os.path.abspath(os.path.join(root, filename.strip('/\\')))
    headers = dict()

    if not filename.startswith(root):
        return HTTPError(403, "Access denied.")
    if not os.path.exists(filename) or not os.path.isfile(filename):
        return HTTPError(404, "File does not exist.")
    if not os.access(filename, os.R_OK):
        return HTTPError(403, "You do not have permission to access this file.")

    if mimetype == 'auto':
        mimetype, encoding = mimetypes.guess_type(filename)
        if encoding: headers['Content-Encoding'] = encoding

    if mimetype:
        if mimetype[:5] == 'text/' and charset and 'charset' not in mimetype:
            mimetype += '; charset=%s' % charset
        headers['Content-Type'] = mimetype

    if download:
        download = os.path.basename(filename if download == True else download)
        headers['Content-Disposition'] = 'attachment; filename="%s"' % download

    stats = os.stat(filename)
    headers['Content-Length'] = clen = stats.st_size
    lm = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stats.st_mtime))
    headers['Last-Modified'] = lm

    ims = request.environ.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims = parse_date(ims.split(";")[0].strip())
    if ims is not None and ims >= int(stats.st_mtime):
        headers['Date'] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        return HTTPResponse(status=304, **headers)

    body = '' if request.method == 'HEAD' else open(filename, 'rb')

    headers["Accept-Ranges"] = "bytes"
    ranges = request.environ.get('HTTP_RANGE')
    if 'HTTP_RANGE' in request.environ:
        ranges = list(parse_range_header(request.environ['HTTP_RANGE'], clen))
        if not ranges:
            return HTTPError(416, "Requested Range Not Satisfiable")
        offset, end = ranges[0]
        headers["Content-Range"] = "bytes %d-%d/%d" % (offset, end-1, clen)
        headers["Content-Length"] = str(end-offset)
        if body: body = _file_iter_range(body, offset, end-offset)
        return HTTPResponse(body, status=206, **headers)
    return HTTPResponse(body, **headers)






###############################################################################
# HTTP Utilities and MISC (TODO) ###############################################
###############################################################################


def debug(mode=True):
    """ Change the debug level.
    There is only one debug level supported at the moment."""
    global DEBUG
    if mode: warnings.simplefilter('default')
    DEBUG = bool(mode)

def http_date(value):
    if isinstance(value, (datedate, datetime)):
        value = value.utctimetuple()
    elif isinstance(value, (int, float)):
        value = time.gmtime(value)
    if not isinstance(value, basestring):
        value = time.strftime("%a, %d %b %Y %H:%M:%S GMT", value)
    return value

def parse_date(ims):
    """ Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. """
    try:
        ts = email.utils.parsedate_tz(ims)
        return time.mktime(ts[:8] + (0,)) - (ts[9] or 0) - time.timezone
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def parse_auth(header):
    """ Parse rfc2617 HTTP authentication header string (basic) and return (user,pass) tuple or None"""
    try:
        method, data = header.split(None, 1)
        if method.lower() == 'basic':
            user, pwd = touni(base64.b64decode(tob(data))).split(':',1)
            return user, pwd
    except (KeyError, ValueError):
        return None

def parse_range_header(header, maxlen=0):
    """ Yield (start, end) ranges parsed from a HTTP Range header. Skip
        unsatisfiable ranges. The end index is non-inclusive."""
    if not header or header[:6] != 'bytes=': return
    ranges = [r.split('-', 1) for r in header[6:].split(',') if '-' in r]
    for start, end in ranges:
        try:
            if not start:  # bytes=-100    -> last 100 bytes
                start, end = max(0, maxlen-int(end)), maxlen
            elif not end:  # bytes=100-    -> all but the first 99 bytes
                start, end = int(start), maxlen
            else:          # bytes=100-200 -> bytes 100-200 (inclusive)
                start, end = int(start), min(int(end)+1, maxlen)
            if 0 <= start < end <= maxlen:
                yield start, end
        except ValueError:
            pass

def _parse_qsl(qs):
    r = []
    for pair in qs.replace(';','&').split('&'):
        if not pair: continue
        nv = pair.split('=', 1)
        if len(nv) != 2: nv.append('')
        key = urlunquote(nv[0].replace('+', ' '))
        value = urlunquote(nv[1].replace('+', ' '))
        r.append((key, value))
    return r

def _lscmp(a, b):
    """ Compares two strings in a cryptographically safe way:
        Runtime is not affected by length of common prefix. """
    return not sum(0 if x==y else 1 for x, y in zip(a, b)) and len(a) == len(b)


def cookie_encode(data, key):
    """ Encode and sign a pickle-able object. Return a (byte) string """
    msg = base64.b64encode(pickle.dumps(data, -1))
    sig = base64.b64encode(hmac.new(tob(key), msg).digest())
    return tob('!') + sig + tob('?') + msg


def cookie_decode(data, key):
    """ Verify and decode an encoded string. Return an object or None."""
    data = tob(data)
    if cookie_is_encoded(data):
        sig, msg = data.split(tob('?'), 1)
        if _lscmp(sig[1:], base64.b64encode(hmac.new(tob(key), msg).digest())):
            return pickle.loads(base64.b64decode(msg))
    return None


def cookie_is_encoded(data):
    """ Return True if the argument looks like a encoded cookie."""
    return bool(data.startswith(tob('!')) and tob('?') in data)




def yieldroutes(func):
    """ Return a generator for routes that match the signature (name, args)
    of the func parameter. This may yield more than one route if the function
    takes optional keyword arguments. The output is best described by example::

        a()         -> '/a'
        b(x, y)     -> '/b/<x>/<y>'
        c(x, y=5)   -> '/c/<x>' and '/c/<x>/<y>'
        d(x=5, y=6) -> '/d' and '/d/<x>' and '/d/<x>/<y>'
    """
    path = '/' + func.__name__.replace('__','/').lstrip('/')
    spec = getargspec(func)
    argc = len(spec[0]) - len(spec[3] or [])
    path += ('/<%s>' * argc) % tuple(spec[0][:argc])
    yield path
    for arg in spec[0][argc:]:
        path += '/<%s>' % arg
        yield path


def path_shift(script_name, path_info, shift=1):
    """ Shift path fragments from PATH_INFO to SCRIPT_NAME and vice versa.

        :return: The modified paths.
        :param script_name: The SCRIPT_NAME path.
        :param script_name: The PATH_INFO path.
        :param shift: The number of path fragments to shift. May be negative to
          change the shift direction. (default: 1)
    """
    if shift == 0: return script_name, path_info
    pathlist = path_info.strip('/').split('/')
    scriptlist = script_name.strip('/').split('/')
    if pathlist and pathlist[0] == '': pathlist = []
    if scriptlist and scriptlist[0] == '': scriptlist = []
    if 0 < shift <= len(pathlist):
        moved = pathlist[:shift]
        scriptlist = scriptlist + moved
        pathlist = pathlist[shift:]
    elif 0 > shift >= -len(scriptlist):
        moved = scriptlist[shift:]
        pathlist = moved + pathlist
        scriptlist = scriptlist[:shift]
    else:
        empty = 'SCRIPT_NAME' if shift < 0 else 'PATH_INFO'
        raise AssertionError("Cannot shift. Nothing left from %s" % empty)
    new_script_name = '/' + '/'.join(scriptlist)
    new_path_info = '/' + '/'.join(pathlist)
    if path_info.endswith('/') and pathlist: new_path_info += '/'
    return new_script_name, new_path_info


def auth_basic(check, realm="private", text="Access denied"):
    """ Callback decorator to require HTTP auth (basic).
        TODO: Add route(check_auth=...) parameter. """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*a, **ka):
            user, password = request.auth or (None, None)
            if user is None or not check(user, password):
                err = HTTPError(401, text)
                err.add_header('WWW-Authenticate', 'Basic realm="%s"' % realm)
                return err
            return func(*a, **ka)
        return wrapper
    return decorator


# Shortcuts for common Bottle methods.
# They all refer to the current default application.

def make_default_app_wrapper(name):
    """ Return a callable that relays calls to the current default app. """
    @functools.wraps(getattr(Bottle, name))
    def wrapper(*a, **ka):
        return getattr(app(), name)(*a, **ka)
    return wrapper

route     = make_default_app_wrapper('route')
get       = make_default_app_wrapper('get')
post      = make_default_app_wrapper('post')
put       = make_default_app_wrapper('put')
delete    = make_default_app_wrapper('delete')
patch     = make_default_app_wrapper('patch')
error     = make_default_app_wrapper('error')
mount     = make_default_app_wrapper('mount')
hook      = make_default_app_wrapper('hook')
install   = make_default_app_wrapper('install')
uninstall = make_default_app_wrapper('uninstall')
url       = make_default_app_wrapper('get_url')








from adapter import server_names, ServerAdapter





###############################################################################
# Application Control ##########################################################
###############################################################################


def load(target, **namespace):
    """ Import a module or fetch an object from a module.

        * ``package.module`` returns `module` as a module object.
        * ``pack.mod:name`` returns the module variable `name` from `pack.mod`.
        * ``pack.mod:func()`` calls `pack.mod.func()` and returns the result.

        The last form accepts not only function calls, but any type of
        expression. Keyword arguments passed to this function are available as
        local variables. Example: ``import_string('re:compile(x)', x='[a-z]')``
    """
    module, target = target.split(":", 1) if ':' in target else (target, None)
    if module not in sys.modules: __import__(module)
    if not target: return sys.modules[module]
    if target.isalnum(): return getattr(sys.modules[module], target)
    package_name = module.split('.')[0]
    namespace[package_name] = sys.modules[package_name]
    return eval('%s.%s' % (module, target), namespace)


def load_app(target):
    """ Load a bottle application from a module and make sure that the import
        does not affect the current default application, but returns a separate
        application object. See :func:`load` for the target parameter. """
    global NORUN; NORUN, nr_old = True, NORUN
    tmp = default_app.push() # Create a new "default application"
    try:
        rv = load(target) # Import the target module
        return rv if callable(rv) else tmp
    finally:
        default_app.remove(tmp) # Remove the temporary added default application
        NORUN = nr_old

_debug = debug
def run(app=None, server='wsgiref', host='127.0.0.1', port=8080,
        interval=1, reloader=False, quiet=False, plugins=None,
        debug=None, **kargs):
    """ Start a server instance. This method blocks until the server terminates.

        :param app: WSGI application or target string supported by
               :func:`load_app`. (default: :func:`default_app`)
        :param server: Server adapter to use. See :data:`server_names` keys
               for valid names or pass a :class:`ServerAdapter` subclass.
               (default: `wsgiref`)
        :param host: Server address to bind to. Pass ``0.0.0.0`` to listens on
               all interfaces including the external one. (default: 127.0.0.1)
        :param port: Server port to bind to. Values below 1024 require root
               privileges. (default: 8080)
        :param reloader: Start auto-reloading server? (default: False)
        :param interval: Auto-reloader interval in seconds (default: 1)
        :param quiet: Suppress output to stdout and stderr? (default: False)
        :param options: Options passed to the server adapter.
     """
    if NORUN: return
    if reloader and not os.environ.get('BOTTLE_CHILD'):
        lockfile = None
        try:
            fd, lockfile = tempfile.mkstemp(prefix='bottle.', suffix='.lock')
            os.close(fd) # We only need this file to exist. We never write to it
            while os.path.exists(lockfile):
                args = [sys.executable] + sys.argv
                environ = os.environ.copy()
                environ['BOTTLE_CHILD'] = 'true'
                environ['BOTTLE_LOCKFILE'] = lockfile
                p = subprocess.Popen(args, env=environ)
                while p.poll() is None: # Busy wait...
                    os.utime(lockfile, None) # I am alive!
                    time.sleep(interval)
                if p.poll() != 3:
                    if os.path.exists(lockfile): os.unlink(lockfile)
                    sys.exit(p.poll())
        except KeyboardInterrupt:
            pass
        finally:
            if os.path.exists(lockfile):
                os.unlink(lockfile)
        return

    try:
        if debug is not None: _debug(debug)
        app = app or default_app()
        if isinstance(app, basestring):
            app = load_app(app)
        if not callable(app):
            raise ValueError("Application is not callable: %r" % app)

        for plugin in plugins or []:
            if isinstance(plugin, basestring):
                plugin = load(plugin)
            app.install(plugin)

        if server in server_names:
            server = server_names.get(server)
        if isinstance(server, basestring):
            server = load(server)
        if isinstance(server, type):
            server = server(host=host, port=port, **kargs)
        if not isinstance(server, ServerAdapter):
            raise ValueError("Unknown or unsupported server: %r" % server)

        server.quiet = server.quiet or quiet
        if not server.quiet:
            _stderr("Bottle v%s server starting up (using %s)...\n" % (__version__, repr(server)))
            _stderr("Listening on http://%s:%d/\n" % (server.host, server.port))
            _stderr("Hit Ctrl-C to quit.\n\n")

        if reloader:
            lockfile = os.environ.get('BOTTLE_LOCKFILE')
            bgcheck = FileCheckerThread(lockfile, interval)
            with bgcheck:
                server.run(app)
            if bgcheck.status == 'reload':
                sys.exit(3)
        else:
            server.run(app)
    except KeyboardInterrupt:
        pass
    except (SystemExit, MemoryError):
        raise
    except:
        if not reloader: raise
        if not getattr(server, 'quiet', quiet):
            print_exc()
        time.sleep(interval)
        sys.exit(3)



class FileCheckerThread(threading.Thread):
    """ Interrupt main-thread as soon as a changed module file is detected,
        the lockfile gets deleted or gets to old. """

    def __init__(self, lockfile, interval):
        threading.Thread.__init__(self)
        self.daemon = True
        self.lockfile, self.interval = lockfile, interval
        #: Is one of 'reload', 'error' or 'exit'
        self.status = None

    def run(self):
        exists = os.path.exists
        mtime = lambda p: os.stat(p).st_mtime
        files = dict()

        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', '')
            if path[-4:] in ('.pyo', '.pyc'): path = path[:-1]
            if path and exists(path): files[path] = mtime(path)

        while not self.status:
            if not exists(self.lockfile)\
            or mtime(self.lockfile) < time.time() - self.interval - 5:
                self.status = 'error'
                thread.interrupt_main()
            for path, lmtime in list(files.items()):
                if not exists(path) or mtime(path) > lmtime:
                    self.status = 'reload'
                    thread.interrupt_main()
                    break
            time.sleep(self.interval)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, *_):
        if not self.status: self.status = 'exit' # silent exit
        self.join()
        return exc_type is not None and issubclass(exc_type, KeyboardInterrupt)





from template import template







###############################################################################
# Constants and Globals ########################################################
###############################################################################


TEMPLATE_PATH = ['./', './views/']
TEMPLATES = {}
DEBUG = False
NORUN = False # If set, run() does nothing. Used by load_app()



#: The default template used for error pages. Override with @error()
ERROR_PAGE_TEMPLATE = """
%%try:
    %%from %s import DEBUG, request
    <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
    <html>
        <head>
            <title>Error: {{e.status}}</title>
            <style type="text/css">
              html {background-color: #eee; font-family: sans-serif;}
              body {background-color: #fff; border: 1px solid #ddd;
                    padding: 15px; margin: 15px;}
              pre {background-color: #eee; border: 1px solid #ddd; padding: 5px;}
            </style>
        </head>
        <body>
            <h1>Error: {{e.status}}</h1>
            <p>Sorry, the requested URL <tt>{{repr(request.url)}}</tt>
               caused an error:</p>
            <pre>{{e.body}}</pre>
            %%if DEBUG and e.exception:
              <h2>Exception:</h2>
              <pre>{{repr(e.exception)}}</pre>
            %%end
            %%if DEBUG and e.traceback:
              <h2>Traceback:</h2>
              <pre>{{e.traceback}}</pre>
            %%end
        </body>
    </html>
%%except ImportError:
    <b>ImportError:</b> Could not generate the error page. Please add bottle to
    the import path.
%%end
""" % __name__

#: A thread-safe instance of :class:`LocalRequest`. If accessed from within a
#: request callback, this instance always refers to the *current* request
#: (even on a multithreaded server).
request = LocalRequest()

#: A thread-safe instance of :class:`LocalResponse`. It is used to change the
#: HTTP response for the *current* request.
response = LocalResponse()

#: A thread-safe namespace. Not used by Bottle.
local = threading.local()

# Initialize app stack (create first empty Bottle app)
# BC: 0.6.4 and needed for run()
app = default_app = AppStack()
app.push()

#: A virtual package that redirects import statements.
#: Example: ``import bottle.ext.sqlite`` actually imports `bottle_sqlite`.
ext = _ImportRedirect('bottle.ext' if __name__ == '__main__' else __name__+".ext", 'bottle_%s').module

if __name__ == '__main__':
    opt, args, parser = _cmd_options, _cmd_args, _cmd_parser
    if opt.version:
        _stdout('Bottle %s\n'%__version__)
        sys.exit(0)
    if not args:
        parser.print_help()
        _stderr('\nError: No application entry point specified.\n')
        sys.exit(1)

    sys.path.insert(0, '.')
    sys.modules.setdefault('bottle', sys.modules['__main__'])

    host, port = (opt.bind or 'localhost'), 8080
    if ':' in host and host.rfind(']') < host.rfind(':'):
        host, port = host.rsplit(':', 1)
    host = host.strip('[]')

    run(args[0], host=host, port=int(port), server=opt.server,
        reloader=opt.reload, plugins=opt.plugin, debug=opt.debug)




# THE END
