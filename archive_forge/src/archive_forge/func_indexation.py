import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
@specs.parameter('obj', Yaqlized(can_index=True))
@specs.name('#indexer')
def indexation(obj, key):
    """:yaql:operator indexer

    Returns value of attribute/property key of the object.

    :signature: obj[key]
    :arg obj: yaqlized object
    :argType obj: yaqlized object, initialized with
        yaqlize_indexer equal to True
    :arg key: index name
    :argType key: keyword
    :returnType: any
    """
    settings = yaqlization.get_yaqlization_settings(obj)
    _validate_name(key, settings, KeyError)
    res = obj[key]
    _auto_yaqlize(res, settings)
    return res