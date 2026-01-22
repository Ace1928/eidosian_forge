import functools
@property
def with_prefixlen(self):
    return '%s/%s' % (self._string_from_ip_int(self._ip), self._prefixlen)