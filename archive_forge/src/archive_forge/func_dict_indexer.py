import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('#indexer')
def dict_indexer(d, key):
    """:yaql:operator indexer

    Returns value of a dictionary by given key.

    :signature: left[right]
    :arg left: input dictionary
    :argType left: dictionary
    :arg right: key
    :argType right: keyword
    :returnType: any (appropriate value type)

    .. code::

        yaql> {"a" => 1, "b" => 2}["a"]
        1
    """
    return d[key]