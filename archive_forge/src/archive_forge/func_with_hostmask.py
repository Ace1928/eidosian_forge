import functools
@property
def with_hostmask(self):
    return '%s/%s' % (self._string_from_ip_int(self._ip), self.hostmask)