import sys
def optionalcascade(cont_attr, item_attr, doc=''):
    """Return a getter property with a cascading setter.

    This is used for the ``id`` and ``description`` properties of the container
    objects with zero or more items. These items have their own private
    attributes that stores query and/or hit ID and description. When the
    container has zero items, attribute values are always retrieved from the
    container's attribute. Otherwise, the first item's attribute is used.

    To keep the container items' query and/or hit ID and description in-sync,
    the setter cascades any new value given to the items' values.

    """

    def getter(self):
        if self._items:
            return getattr(self[0], item_attr)
        else:
            return getattr(self, cont_attr)

    def setter(self, value):
        setattr(self, cont_attr, value)
        for item in self:
            setattr(item, item_attr, value)
    return property(fget=getter, fset=setter, doc=doc)