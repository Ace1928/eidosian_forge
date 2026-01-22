import functools
@property
@functools.lru_cache()
def is_private(self):
    """Test if this address is allocated for private networks.

        Returns:
            A boolean, True if the address is reserved per
            iana-ipv6-special-registry, or is ipv4_mapped and is
            reserved in the iana-ipv4-special-registry.

        """
    ipv4_mapped = self.ipv4_mapped
    if ipv4_mapped is not None:
        return ipv4_mapped.is_private
    return any((self in net for net in self._constants._private_networks))