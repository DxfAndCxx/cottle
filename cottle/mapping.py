# -*- coding:utf-8 -*-
#    author    :   丁雪峰
#    time      :   2014-11-20 10:53:13
#    email     :   fengidri@yeah.net
#    version   :   1.0.1
from http_wsgi import HTTPError
import re
import types
class Mapping(object):
    def load(self, urls, fvars):
        self.mapping = urls
        self.fvars = fvars
    def match(self, path):
        for pat, handle in self.mapping:
            #暂时不支持application
            #webpy中动态修改回调字符串的方式也不支持
            pat = '^%s$' % pat
            match = re.search(pat, path)
            if match:
                return handle, match.groups()
        return None, []
    def call(self, f, args,  request):
            
        if f is None:
            raise HTTPError(404, "Not found")
        elif isinstance(f, (types.ClassType, type)): # is_class
            return self.handle_class(f, args, request)
        elif isinstance(f, basestring):
            cls = None
            if '.' in f:
                mod, cls = f.rsplit('.', 1)
                mod = __import__(mod, None, None, [''])
                cls = getattr(mod, cls)
            else:
                cls = self.fvars.get(f)
            if cls:
                return self.handle_class(cls, args, request)
            raise HTTPError(404, "Not found: %s" % f)
        elif hasattr(f, '__call__'):
            return f(*args)
        else:
            raise HTTPError(404, "Not found: %s" % f)

    def handle_class(self, cls, args, request):
        meth = request.method
        if meth == 'HEAD' and not hasattr(cls, meth):
            meth = 'GET'
        if not hasattr(cls, meth):
            raise HTTPError(404, "Not Found Handle:%s" % meth)
        chandle = cls()

        chandle.request = request
        if hasattr(chandle, 'Before'):
            if getattr(chandle, 'Before')():
                res = getattr(chandle, meth)(*args)
            else:
                return ''#stop by Before. some Exception will be raise in Before
        else:
            res = getattr(chandle, meth)(*args)

        if hasattr(chandle, 'After'):
            getattr(chandle, 'After')()
        return res

if __name__ == "__main__":
    pass

