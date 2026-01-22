import weakref, traceback, sys
def calculateKey(cls, target):
    """Calculate the reference key for this reference

        Currently this is a two-tuple of the id()'s of the
        target object and the target function respectively.
        """
    return (id(getattr(target, im_self)), id(getattr(target, im_func)))