def set_repr(self):
    if len(self) == 0:
        return 'set()'
    lst = [repr(x) for x in self]
    return 'set([' + ', '.join(lst) + '])'