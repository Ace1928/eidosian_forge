import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('left', int)
@specs.parameter('right', yaqltypes.Sequence())
@specs.name('#operator_*')
def int_by_list(left, right, engine):
    """:yaql:operator *

    Returns sequence repeated count times.

    :signature: left * right
    :arg left: multiplier
    :argType left: integer
    :arg right: input sequence
    :argType right: sequence
    :returnType: sequence

    .. code::

        yaql> 2 * [1, 2]
        [1, 2, 1, 2]
    """
    return list_by_int(right, left, engine)