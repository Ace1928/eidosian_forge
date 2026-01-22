def when_type(*types):
    """Decorator to add a method that will be called for the given types"""
    for t in types:
        if not isinstance(t, classtypes):
            raise TypeError('%r is not a type or class' % (t,))

    def decorate(f):
        for t in types:
            if _by_type.setdefault(t, f) is not f:
                raise TypeError('%r already has method for type %r' % (func, t))
        return f
    return decorate