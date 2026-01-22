import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('left', yaqltypes.Number())
@specs.parameter('right', yaqltypes.Number())
@specs.name('#operator_mod')
def modulo(left, right):
    """:yaql:operator mod

    Returns left modulo right.

    :signature: left mod right
    :arg left: left operand
    :argType left: number
    :arg right: right operand
    :argType right: number
    :returnType: number

    .. code::

        yaql> 3 mod 2
        1
    """
    return left % right