import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('sequence', yaqltypes.Iterable())
@specs.parameter('separator', yaqltypes.String())
@specs.inject('str_delegate', yaqltypes.Delegate('str'))
@specs.method
def join_(separator, sequence, str_delegate):
    """:yaql:join

    Returns a string with sequence elements joined by the separator.

    :signature: separator.join(sequence)
    :receiverArg separator: value to be placed between joined pairs
    :argType separator: string
    :arg sequence: chain of values to be joined
    :argType sequence: sequence of strings
    :returnType: string

    .. code::

        yaql> "|".join(["abc", "de", "f"])
        "abc|de|f"
    """
    return join(sequence, separator, str_delegate)