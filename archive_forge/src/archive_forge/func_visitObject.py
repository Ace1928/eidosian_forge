import enum
def visitObject(self, obj, *args, **kwargs):
    """Called to visit an object. This function loops over all non-private
        attributes of the objects and calls any user-registered (via
        @register_attr() or @register_attrs()) visit() functions.

        If there is no user-registered visit function, of if there is and it
        returns True, or it returns None (or doesn't return anything) and
        visitor.defaultStop is False (default), then the visitor will proceed
        to call self.visitAttr()"""
    keys = sorted(vars(obj).keys())
    _visitors = self._visitorsFor(obj)
    defaultVisitor = _visitors.get('*', None)
    for key in keys:
        if key[0] == '_':
            continue
        value = getattr(obj, key)
        visitorFunc = _visitors.get(key, defaultVisitor)
        if visitorFunc is not None:
            ret = visitorFunc(self, obj, key, value, *args, **kwargs)
            if ret == False or (ret is None and self.defaultStop):
                continue
        self.visitAttr(obj, key, value, *args, **kwargs)