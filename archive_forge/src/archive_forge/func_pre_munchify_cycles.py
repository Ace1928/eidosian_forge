from collections.abc import Mapping
def pre_munchify_cycles(obj):
    try:
        return (seen[id(obj)], True)
    except KeyError:
        pass
    seen[id(obj)] = partial = pre_munchify(obj)
    return (partial, False)