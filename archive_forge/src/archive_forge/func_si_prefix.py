import collections
def si_prefix(self, value):
    self._validate_named(value, Prefix)
    self._prefix = value
    return self