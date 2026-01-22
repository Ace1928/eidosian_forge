from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
def validateSetattr(obj, name, value):
    """validate setattr(obj,name,value)"""
    if rl_config.shapeChecking:
        aMap = obj._attrMap
        if aMap and name[0] != '_':
            if isinstance(value, DerivedValue):
                pass
            elif name in aMap:
                validate = aMap[name].validate
                try:
                    r = validate(value)
                except Exception as e:
                    raise e.__class__(f'{obj.__class__.__name__}.{name} {validate}({value!r})') from e
                else:
                    if not r:
                        raise AttributeError(f'Illegal assignment of {value!r} to {name} in class {obj.__class__.__name__}')
            else:
                prop = getattr(obj.__class__, name, None)
                if isinstance(prop, property):
                    fset = getattr(prop, 'fset', None)
                    if fset:
                        fset(obj, value)
                        return
                    else:
                        raise AttributeError(f'{obj.__class__.__name__}.{name} has no setter')
                else:
                    raise AttributeError("Illegal attribute '%s' in class %s" % (name, obj.__class__.__name__))
    prop = getattr(obj.__class__, name, None)
    if isinstance(prop, property):
        fset = getattr(prop, 'fset', None)
        if fset:
            fset(obj, value)
        else:
            raise AttributeError(f'{obj.__class__.__name__}.{name} has no setter')
    elif name == '__dict__':
        obj.__dict__.clear()
        obj.__dict__.update(value)
    else:
        obj.__dict__[name] = value