import functools
@property
def scope_id(self):
    """Identifier of a particular zone of the address's scope.

        See RFC 4007 for details.

        Returns:
            A string identifying the zone of the address if specified, else None.

        """
    return self._scope_id