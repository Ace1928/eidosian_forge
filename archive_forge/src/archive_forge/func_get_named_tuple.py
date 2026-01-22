import collections
def get_named_tuple(name, dict):
    return collections.namedtuple(name, dict.keys())(*dict.values())