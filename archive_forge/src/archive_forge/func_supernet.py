import functools
def supernet(self, prefixlen_diff=1, new_prefix=None):
    """The supernet containing the current network.

        Args:
            prefixlen_diff: An integer, the amount the prefix length of
              the network should be decreased by.  For example, given a
              /24 network and a prefixlen_diff of 3, a supernet with a
              /21 netmask is returned.

        Returns:
            An IPv4 network object.

        Raises:
            ValueError: If self.prefixlen - prefixlen_diff < 0. I.e., you have
              a negative prefix length.
                OR
            If prefixlen_diff and new_prefix are both set or new_prefix is a
              larger number than the current prefix (larger number means a
              smaller network)

        """
    if self._prefixlen == 0:
        return self
    if new_prefix is not None:
        if new_prefix > self._prefixlen:
            raise ValueError('new prefix must be shorter')
        if prefixlen_diff != 1:
            raise ValueError('cannot set prefixlen_diff and new_prefix')
        prefixlen_diff = self._prefixlen - new_prefix
    new_prefixlen = self.prefixlen - prefixlen_diff
    if new_prefixlen < 0:
        raise ValueError('current prefixlen is %d, cannot have a prefixlen_diff of %d' % (self.prefixlen, prefixlen_diff))
    return self.__class__((int(self.network_address) & int(self.netmask) << prefixlen_diff, new_prefixlen))