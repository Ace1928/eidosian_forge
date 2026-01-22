from decimal import Decimal
from boto.compat import filter, map
def search_scopes(self, key):
    for scope in self.scopes:
        if hasattr(scope, key):
            return getattr(scope, key)
        if hasattr(scope, '__getitem__'):
            if key in scope:
                return scope[key]