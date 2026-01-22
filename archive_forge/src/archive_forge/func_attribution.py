import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
@specs.parameter('obj', Yaqlized(can_access_attributes=True))
@specs.parameter('attr', yaqltypes.Keyword())
@specs.name('#operator_.')
def attribution(obj, attr):
    """:yaql:operator .

    Returns attribute of the object.

    :signature: obj.attr
    :arg obj: yaqlized object
    :argType obj: yaqlized object, initialized with
        yaqlize_attributes equal to True
    :arg attr: attribute name
    :argType attr: keyword
    :returnType: any
    """
    settings = yaqlization.get_yaqlization_settings(obj)
    _validate_name(attr, settings)
    attr = _remap_name(attr, settings)
    res = getattr(obj, attr)
    _auto_yaqlize(res, settings)
    return res