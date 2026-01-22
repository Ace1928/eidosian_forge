import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.inject('__context__', yaqltypes.Context())
def let(__context__, *args, **kwargs):
    """:yaql:let

    Returns context object where args are stored with 1-based indexes
    and kwargs values are stored with appropriate keys.

    :signature: let([args], {kwargs})
    :arg [args]: values to be stored under appropriate numbers $1, $2, ...
    :argType [args]: chain of any values
    :arg {kwargs}: values to be stored under appropriate keys
    :argType {kwargs}: chain of mappings
    :returnType: context object

    .. code::

        yaql> let(1, 2, a => 3, b => 4) -> $1 + $a + $2 + $b
        10
    """
    for i, value in enumerate(args, 1):
        __context__[str(i)] = value
    for key, value in kwargs.items():
        __context__[key] = value
    return __context__