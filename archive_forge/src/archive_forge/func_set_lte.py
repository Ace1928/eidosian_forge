import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.parameter('left', utils.SetType)
@specs.parameter('right', utils.SetType)
@specs.name('#operator_<=')
def set_lte(left, right):
    """:yaql:operator <=

    Returns true if left set is subset of right set.

    :signature: left <= right
    :arg left: left set
    :argType left: set
    :arg right: right set
    :argType right: set
    :returnType: boolean

    .. code::

        yaql> set(0, 1) <= set(0, 1)
        true
    """
    return left <= right