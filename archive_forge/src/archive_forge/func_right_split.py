import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.parameter('separator', yaqltypes.String(nullable=True))
@specs.parameter('max_splits', int)
@specs.method
def right_split(string, separator=None, max_splits=-1):
    """:yaql:rightSplit

    Returns a list of tokens in the string, using separator as the
    delimiter. If maxSplits is given then at most maxSplits splits are done -
    the rightmost ones.

    :signature: string.rightSplit(separator => null, maxSplits => -1)
    :receiverArg string: value to be splitted
    :argType string: string
    :arg separator: delimiter for splitting. null by default, which means
        splitting with whitespace characters
    :argType separator: string
    :arg maxSplits: number of splits to be done - the rightmost ones.
        -1 by default, which means all possible splits are done
    :argType maxSplits: integer
    :returnType: list

    .. code::

        yaql> "abc     de  f".rightSplit()
        ["abc", "de", "f"]
        yaql> "abc     de  f".rightSplit(maxSplits => 1)
        ["abc     de", "f"]
    """
    return string.rsplit(separator, max_splits)