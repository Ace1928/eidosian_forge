import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('#indexer')
def dict_indexer_with_default(d, key, default):
    """:yaql:operator indexer

    Returns value of a dictionary by given key or default if there is
    no such key.

    :signature: left[right, default]
    :arg left: input dictionary
    :argType left: dictionary
    :arg right: key
    :argType right: keyword
    :arg default: default value to be returned if key is missing in dictionary
    :argType default: any
    :returnType: any (appropriate value type)

    .. code::

        yaql> {"a" => 1, "b" => 2}["c", 3]
        3
    """
    return d.get(key, default)