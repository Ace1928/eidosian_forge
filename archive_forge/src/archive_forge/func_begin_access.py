import sys
def begin_access(self, cursor=None, offset=0, size=sys.maxsize, flags=0):
    """Call this before the first use of this instance. The method was already
        called by the constructor in case sufficient information was provided.

        For more information no the parameters, see the __init__ method
        :param path: if cursor is None the existing one will be used.
        :return: True if the buffer can be used"""
    if cursor:
        self._c = cursor
    if self._c is not None and self._c.is_associated():
        res = self._c.use_region(offset, size, flags).is_valid()
        if res:
            if size > self._c.file_size():
                size = self._c.file_size() - offset
            self._size = size
        return res
    return False