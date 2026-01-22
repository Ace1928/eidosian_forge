import ctypes
from OpenGL import plugins
@classmethod
def typeLookup(cls, type):
    """Lookup handler by data-type"""
    registry = ArrayDatatype.getRegistry()
    try:
        return registry[type]
    except KeyError as err:
        key = '%s.%s' % (type.__module__, type.__name__)
        plugin = cls.LAZY_TYPE_REGISTRY.get(key)
        if plugin:
            cls.loadPlugin(plugin)
            return registry[type]
        raise KeyError('Unable to find data-format handler for %s' % (type,))