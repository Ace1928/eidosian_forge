import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('value', nullable=True)
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.method
def contains_value(d, value):
    """:yaql:containsValue

    Returns true if the dictionary contains the value, false otherwise.

    :signature: dict.containsValue(value)
    :receiverArg dict: dictionary to find occurrence in
    :argType dict: mapping
    :arg value: value to be checked for occurrence
    :argType value: any
    :returnType: boolean

    .. code::

        yaql> {"a" => 1, "b" => 2}.containsValue("a")
        false
        yaql> {"a" => 1, "b" => 2}.containsValue(2)
        true
    """
    return value in d.values()