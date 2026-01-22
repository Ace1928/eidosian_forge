def build_properties(cls, new_attrs):
    """
    Given a class and a new set of attributes (as passed in by
    __classinit__), create or modify properties based on functions
    with special names ending in __get, __set, and __del.
    """
    for name, value in new_attrs.items():
        if name.endswith('__get') or name.endswith('__set') or name.endswith('__del'):
            base = name[:-5]
            if hasattr(cls, base):
                old_prop = getattr(cls, base)
                if not isinstance(old_prop, property):
                    raise ValueError('Attribute %s is a %s, not a property; function %s is named like a property' % (base, type(old_prop), name))
                attrs = {'fget': old_prop.fget, 'fset': old_prop.fset, 'fdel': old_prop.fdel, 'doc': old_prop.__doc__}
            else:
                attrs = {}
            attrs['f' + name[-3:]] = value
            if name.endswith('__get') and value.__doc__:
                attrs['doc'] = value.__doc__
            new_prop = property(**attrs)
            setattr(cls, base, new_prop)