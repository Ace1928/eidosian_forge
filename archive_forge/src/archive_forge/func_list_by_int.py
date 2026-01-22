import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('left', yaqltypes.Sequence())
@specs.parameter('right', int)
@specs.name('#operator_*')
def list_by_int(left, right, engine):
    """:yaql:operator *

    Returns sequence repeated count times.

    :signature: left * right
    :arg left: input sequence
    :argType left: sequence
    :arg right: multiplier
    :argType right: integer
    :returnType: sequence

    .. code::

        yaql> [1, 2] * 2
        [1, 2, 1, 2]
    """
    utils.limit_memory_usage(engine, (-right + 1, []), (right, left))
    return left * right