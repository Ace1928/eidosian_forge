import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.method
def to_char_array(string):
    """:yaql:toCharArray

    Converts a string to array of one character strings.

    :signature: string.toCharArray()
    :receiverArg string: input string
    :argType string: string
    :returnType: list

    .. code::

        yaql> "abc de".toCharArray()
        ["a", "b", "c", " ", "d", "e"]
    """
    return tuple(string)