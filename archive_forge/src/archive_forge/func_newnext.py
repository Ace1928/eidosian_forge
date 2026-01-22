def newnext(iterator, default=_SENTINEL):
    """
    next(iterator[, default])

    Return the next item from the iterator. If default is given and the iterator
    is exhausted, it is returned instead of raising StopIteration.
    """
    try:
        try:
            return iterator.__next__()
        except AttributeError:
            try:
                return iterator.next()
            except AttributeError:
                raise TypeError("'{0}' object is not an iterator".format(iterator.__class__.__name__))
    except StopIteration as e:
        if default is _SENTINEL:
            raise e
        else:
            return default