# -*- coding:utf-8 -*-
#    author    :   丁雪峰
#    time      :   2014-11-20 10:53:13
#    email     :   fengidri@yeah.net
#    version   :   1.0.1
from http_wsgi import HTTPError, HTTPResponse
import re
import types
from template import template as template
from py23k import *


# 用于增加到用户类的方法

@property
def attr_query(self):
    return self.request.query
 
@property
def attr_forms(self):
    return self.request.forms

@property
def attr_path(self):
    return self.request.path

@property
def attr_env(self):
    return self.request.environ

def attr_template(self, *args, **kwargs):
    return template(*args, **kwargs)

def attr_abort(self, code=500, text='Unknown Error.'):
    """ Aborts execution and causes a HTTP error. """
    raise HTTPError(code, text)


def attr_redirect(self, url, code=None):
    """ Aborts execution and causes a 303 or 302 redirect, depending on
        the HTTP protocol version. """
    if not code:
        if self.request.get('SERVER_PROTOCOL') == "HTTP/1.1": 
            code = 303
        else:
            code = 302

    res = response.copy(cls=HTTPResponse)
    res.status = code
    res.body = ""
    res.set_header('Location', urljoin(request.url, url))
    raise res

################################################################################
class Mapping(object):
    def load(self, mapping, fvars):
        self.mapping = []


        for pat, handle in mapping:
            handle = self.init_handle(handle, fvars)
            if not handle: continue
            self.mapping.append((pat, handle))

    def init_handle(self, handle, fvars):
        if handle is None:
            return
        elif isinstance(handle, (types.ClassType, type)): # is_class
            return self.__init_handle(handle)
        elif isinstance(f, basestring):
            cls = None
            if '.' in f:
                mod, cls = f.rsplit('.', 1)
                mod = __import__(mod, None, None, [''])
                cls = getattr(mod, cls)
            else:
                cls = self.fvars.get(f)
            if cls:
                return self.__init_handle(cls)
            return 
        else:
            return

    def __init_handle(self, cls):
        setattr(cls, 'request',  None)
        setattr(cls, 'response', None)
        setattr(cls, 'query',    attr_query)
        setattr(cls, 'forms',    attr_forms)
        setattr(cls, 'path',     attr_path)
        setattr(cls, 'env',      attr_env)
        setattr(cls, 'template', attr_template)
        setattr(cls, 'abort',    attr_abort)
        setattr(cls, 'redirect', attr_redirect)
        return cls()


    def match(self, path):
        for pat, handle in self.mapping:
            #暂时不支持application
            #webpy中动态修改回调字符串的方式也不支持
            pat = '^%s$' % pat
            match = re.search(pat, path)
            if match:
                return handle, match.groups()
        return None, []

    def call(self, handle, args, request, response):
        if handle is None:
            raise HTTPError(404, "Not Found")

        meth = request.method
        if meth == 'HEAD' and not hasattr(handle, meth):
            meth = 'GET'
        if not hasattr(handle, meth):
            raise HTTPError(404, "Not Found Handle:%s" % meth)


        handle.request  = request
        handle.response = response

        if hasattr(handle, 'Before'):# 用户的定义了Before 方法
            if getattr(handle, 'Before')():
                res = getattr(handle, meth)(*args)
            else:
                return ''#stop by Before. some Exception will be raise in Before
        else:
            res = getattr(handle, meth)(*args)

        if hasattr(handle, 'After'):
            getattr(handle, 'After')()

        if not isinstance(res, basestring):
            response.content_type = "application/json"
            return json_dumps(res)
        return res

if __name__ == "__main__":
    pass

