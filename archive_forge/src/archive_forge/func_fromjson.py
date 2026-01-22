import datetime
import decimal
import json
from simplegeneric import generic
import wsme.exc
import wsme.types
from wsme.types import Unset
import wsme.utils
@generic
def fromjson(datatype, value):
    """A generic converter from json base types to python datatype.

    If a non-complex user specific type is to be used in the api,
    a specific fromjson should be added::

        from wsme.protocol.restjson import fromjson

        class MySpecialType(object):
            pass

        @fromjson.when_object(MySpecialType)
        def myspecialtype_fromjson(datatype, value):
            return MySpecialType(value)
    """
    if value is None:
        return None
    if wsme.types.iscomplex(datatype):
        obj = datatype()
        attributes = wsme.types.list_attributes(datatype)
        v_keys = set(value.keys())
        a_keys = set((adef.name for adef in attributes))
        if not v_keys <= a_keys:
            raise wsme.exc.UnknownAttribute(None, v_keys - a_keys)
        for attrdef in attributes:
            if attrdef.name in value:
                try:
                    val_fromjson = fromjson(attrdef.datatype, value[attrdef.name])
                except wsme.exc.UnknownAttribute as e:
                    e.add_fieldname(attrdef.name)
                    raise
                if getattr(attrdef, 'readonly', False):
                    raise wsme.exc.InvalidInput(attrdef.name, val_fromjson, 'Cannot set read only field.')
                setattr(obj, attrdef.key, val_fromjson)
            elif attrdef.mandatory:
                raise wsme.exc.InvalidInput(attrdef.name, None, 'Mandatory field missing.')
        return wsme.types.validate_value(datatype, obj)
    elif wsme.types.isusertype(datatype):
        value = datatype.frombasetype(fromjson(datatype.basetype, value))
    return value