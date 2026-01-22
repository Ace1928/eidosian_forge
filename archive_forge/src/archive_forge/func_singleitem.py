import sys
def singleitem(attr=None, doc=''):
    """Property for fetching attribute from first entry of container.

    Returns a property that fetches the given attribute from
    the first item in a SearchIO container object.
    """

    def getter(self):
        if len(self._items) > 1:
            raise ValueError('More than one HSPFragment objects found in HSP')
        if attr is None:
            return self._items[0]
        return getattr(self._items[0], attr)
    return property(fget=getter, doc=doc)