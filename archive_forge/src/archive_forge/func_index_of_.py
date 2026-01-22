import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.parameter('sub', yaqltypes.String())
@specs.parameter('start', int)
@specs.parameter('length', int)
@specs.method
def index_of_(string, sub, start, length):
    """:yaql:indexOf

    Returns an index of first occurrence sub in string beginning from start
    ending with start+length.
    -1 is a return value if there is no any occurrence.

    :signature: string.indexOf(sub, start, length)
    :receiverArg string: input string
    :argType string: string
    :arg sub: substring to find in string
    :argType sub: string
    :arg start: index to start search with, 0 by default
    :argType start: integer
    :arg length: length of string to find substring in
    :argType length: integer
    :returnType: integer

    .. code::

        yaql> "cabcdab".indexOf("bc", 2, 2)
        2
    """
    if start < 0:
        start += len(string)
    if length < 0:
        length = len(string) - start
    return string.find(sub, start, start + length)