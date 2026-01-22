import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
def with_(context, *args):
    """:yaql:with

    Returns new context object where args are stored with 1-based indexes.

    :signature: with([args])
    :arg [args]: values to be stored under appropriate numbers $1, $2, ...
    :argType [args]: chain of any values
    :returnType: context object

    .. code::

        yaql> with("ab", "cd") -> $1 + $2
        "abcd"
    """
    for i, t in enumerate(args, 1):
        context[str(i)] = t
    return context