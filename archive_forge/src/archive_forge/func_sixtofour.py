import functools
@property
def sixtofour(self):
    """Return the IPv4 6to4 embedded address.

        Returns:
            The IPv4 6to4-embedded address if present or None if the
            address doesn't appear to contain a 6to4 embedded address.

        """
    if self._ip >> 112 != 8194:
        return None
    return IPv4Address(self._ip >> 80 & 4294967295)