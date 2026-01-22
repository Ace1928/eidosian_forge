from decimal import Decimal
from boto.compat import filter, map
def strip_namespace(func):

    def wrapper(self, name, *args, **kw):
        if self._namespace is not None:
            if name.startswith(self._namespace + ':'):
                name = name[len(self._namespace + ':'):]
        return func(self, name, *args, **kw)
    return wrapper