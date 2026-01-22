from .python3_compat import iterkeys, iteritems, Mapping  #, u
def pre_munchify(obj):
    if isinstance(obj, Mapping):
        return factory({})
    elif isinstance(obj, list):
        return type(obj)()
    elif isinstance(obj, tuple):
        type_factory = getattr(obj, '_make', type(obj))
        return type_factory((munchify_cycles(item) for item in obj))
    else:
        return obj