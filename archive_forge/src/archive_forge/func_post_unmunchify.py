from .python3_compat import iterkeys, iteritems, Mapping  #, u
def post_unmunchify(partial, obj):
    if isinstance(obj, Mapping):
        partial.update(((k, unmunchify_cycles(obj[k])) for k in iterkeys(obj)))
    elif isinstance(obj, list):
        partial.extend((unmunchify_cycles(v) for v in obj))
    elif isinstance(obj, tuple):
        for value_partial, value in zip(partial, obj):
            post_unmunchify(value_partial, value)
    return partial