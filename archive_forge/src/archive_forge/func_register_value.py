from __future__ import unicode_literals
@classmethod
def register_value(cls, value, obj):
    assert cls._id_map is not EnumObject._id_map
    if value in cls._id_map:
        raise ValueError('{0} value {1} already loaded'.format(cls.__name__, value))
    cls._id_map[value] = obj