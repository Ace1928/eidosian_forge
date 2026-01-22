import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.name('values')
@specs.method
def dict_values(d):
    """:yaql:values

    Returns an iterator over the dictionary values.

    :signature: dict.values()
    :receiverArg dict: input dictionary
    :argType dict: dictionary
    :returnType: iterator

    .. code::

        yaql> {"a" => 1, "b" => 2}.values()
        [1, 2]
    """
    return d.values()