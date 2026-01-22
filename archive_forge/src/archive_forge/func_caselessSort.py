import ast
import string
def caselessSort(alist):
    """Return a sorted copy of a list. If there are only strings
    in the list, it will not consider case.
    """
    try:
        return sorted(alist, key=lambda a: (a.lower(), a))
    except TypeError:
        return sorted(alist)