def when_object(*obs):
    """Decorator to add a method to be called for the given object(s)"""

    def decorate(f):
        for o in obs:
            if _by_object.setdefault(id(o), (o, f))[1] is not f:
                raise TypeError('%r already has method for object %r' % (func, o))
        return f
    return decorate