from .python3_compat import iterkeys, iteritems, Mapping  #, u
def munchify_cycles(obj):
    try:
        return seen[id(obj)]
    except KeyError:
        pass
    seen[id(obj)] = partial = pre_munchify(obj)
    return post_munchify(partial, obj)