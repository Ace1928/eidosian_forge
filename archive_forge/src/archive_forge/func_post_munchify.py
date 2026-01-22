from .python3_compat import iterkeys, iteritems, Mapping  #, u
def post_munchify(partial, obj):
    if isinstance(obj, Mapping):
        partial.update(((k, munchify_cycles(obj[k])) for k in iterkeys(obj)))
    elif isinstance(obj, list):
        partial.extend((munchify_cycles(item) for item in obj))
    elif isinstance(obj, tuple):
        for item_partial, item in zip(partial, obj):
            post_munchify(item_partial, item)
    return partial