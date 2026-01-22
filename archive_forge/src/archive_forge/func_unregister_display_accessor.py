import weakref
def unregister_display_accessor(name):
    if name not in _display_accessors:
        raise KeyError('No such display accessor: {name!r}')
    del _display_accessors[name]
    for fn in _reactive_display_objs:
        delattr(fn, name)