import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('value', int)
@specs.parameter('bits_number', int)
def shift_bits_right(value, bits_number):
    """:yaql:shiftBitsRight

    Shifts the bits of value right by the number of bits bitsNumber.

    :signature: shiftBitsRight(value, bitsNumber)
    :arg value: given value
    :argType value: integer
    :arg bitsNumber: number of bits
    :argType right: integer
    :returnType: integer

    .. code::

        yaql> shiftBitsRight(8, 2)
        2
    """
    return value >> bits_number