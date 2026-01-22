def is_closable_iterator(obj):
    """Detect if the given object is both closable and iterator."""
    if not is_iterator(obj):
        return False
    import inspect
    if inspect.isgenerator(obj):
        return True
    if not (hasattr(obj, 'close') and callable(obj.close)):
        return False
    try:
        inspect.getcallargs(obj.close)
    except TypeError:
        return False
    else:
        return True