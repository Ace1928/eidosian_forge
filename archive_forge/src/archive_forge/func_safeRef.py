import weakref, traceback, sys
def safeRef(target, onDelete=None):
    """Return a *safe* weak reference to a callable target

    target -- the object to be weakly referenced, if it's a
        bound method reference, will create a BoundMethodWeakref,
        otherwise creates a simple weakref.
    onDelete -- if provided, will have a hard reference stored
        to the callable to be called after the safe reference
        goes out of scope with the reference object, (either a
        weakref or a BoundMethodWeakref) as argument.
    """
    if hasattr(target, im_self):
        if getattr(target, im_self) is not None:
            assert hasattr(target, im_func), "safeRef target %r has %s, but no %s, don't know how to create reference" % (target, im_self, im_func)
            reference = BoundMethodWeakref(target=target, onDelete=onDelete)
            return reference
    if onDelete is not None:
        return weakref.ref(target, onDelete)
    else:
        return weakref.ref(target)