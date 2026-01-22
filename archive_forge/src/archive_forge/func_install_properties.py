from . import _gi
from ._constants import \
def install_properties(cls):
    """
    Scans the given class for instances of Property and merges them
    into the classes __gproperties__ dict if it exists or adds it if not.
    """
    gproperties = cls.__dict__.get('__gproperties__', {})
    props = []
    for name, prop in cls.__dict__.items():
        if isinstance(prop, Property):
            if not prop.name:
                prop.name = name
            if prop.name in gproperties:
                if gproperties[prop.name] == prop.get_pspec_args():
                    continue
                raise ValueError('Property %s was already found in __gproperties__' % prop.name)
            gproperties[prop.name] = prop.get_pspec_args()
            props.append(prop)
    if not props:
        return
    cls.__gproperties__ = gproperties
    if 'do_get_property' in cls.__dict__ or 'do_set_property' in cls.__dict__:
        for prop in props:
            if prop.fget != prop._default_getter or prop.fset != prop._default_setter:
                raise TypeError('GObject subclass %r defines do_get/set_property and it also uses a property with a custom setter or getter. This is not allowed' % (cls.__name__,))

    def obj_get_property(self, pspec):
        name = pspec.name.replace('-', '_')
        return getattr(self, name, None)
    cls.do_get_property = obj_get_property

    def obj_set_property(self, pspec, value):
        name = pspec.name.replace('-', '_')
        prop = getattr(cls, name, None)
        if prop:
            prop.fset(self, value)
    cls.do_set_property = obj_set_property