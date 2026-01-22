import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
@classmethod
def from_legacy(cls, schema_dict):
    """Return a Property Schema object from a legacy schema dictionary."""
    if isinstance(schema_dict, cls):
        return schema_dict
    unknown = [k for k in schema_dict if k not in SCHEMA_KEYS]
    if unknown:
        raise exception.InvalidSchemaError(message=_('Unknown key(s) %s') % unknown)

    def constraints():

        def get_num(key):
            val = schema_dict.get(key)
            if val is not None:
                val = Schema.str_to_num(val)
            return val
        if MIN_VALUE in schema_dict or MAX_VALUE in schema_dict:
            yield constr.Range(get_num(MIN_VALUE), get_num(MAX_VALUE))
        if MIN_LENGTH in schema_dict or MAX_LENGTH in schema_dict:
            yield constr.Length(get_num(MIN_LENGTH), get_num(MAX_LENGTH))
        if ALLOWED_VALUES in schema_dict:
            yield constr.AllowedValues(schema_dict[ALLOWED_VALUES])
        if ALLOWED_PATTERN in schema_dict:
            yield constr.AllowedPattern(schema_dict[ALLOWED_PATTERN])
    try:
        data_type = schema_dict[TYPE]
    except KeyError:
        raise exception.InvalidSchemaError(message=_('No %s specified') % TYPE)
    if SCHEMA in schema_dict:
        if data_type == Schema.LIST:
            ss = cls.from_legacy(schema_dict[SCHEMA])
        elif data_type == Schema.MAP:
            schema_dicts = schema_dict[SCHEMA].items()
            ss = dict(((n, cls.from_legacy(sd)) for n, sd in schema_dicts))
        else:
            raise exception.InvalidSchemaError(message=_('%(schema)s supplied for %(type)s %(data)s') % dict(schema=SCHEMA, type=TYPE, data=data_type))
    else:
        ss = None
    return cls(data_type, description=schema_dict.get(DESCRIPTION), default=schema_dict.get(DEFAULT), schema=ss, required=schema_dict.get(REQUIRED, False), constraints=list(constraints()), implemented=schema_dict.get(IMPLEMENTED, True), update_allowed=schema_dict.get(UPDATE_ALLOWED, False), immutable=schema_dict.get(IMMUTABLE, False))