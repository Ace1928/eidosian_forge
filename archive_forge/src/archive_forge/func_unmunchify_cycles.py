from .python3_compat import iterkeys, iteritems, Mapping  #, u
def unmunchify_cycles(obj):
    try:
        return seen[id(obj)]
    except KeyError:
        pass
    seen[id(obj)] = partial = pre_unmunchify(obj)
    return post_unmunchify(partial, obj)