def union_update(self, other):
    """Update the set, adding any elements from other which are not
        already in the set.
        """
    if not isinstance(other, Set):
        raise ValueError('other must be a Set instance')
    if self is other:
        return
    for item in other.items:
        self.add(item)