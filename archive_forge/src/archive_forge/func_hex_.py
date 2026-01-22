import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('num', yaqltypes.Number(nullable=True))
def hex_(num):
    """:yaql:hex

    Returns a string with hexadecimal representation of num.

    :signature: hex(num)
    :arg num: input number to be converted to hexademical
    :argType num: number
    :returnType: string

    .. code::

        yaql> hex(256)
        "0x100"
    """
    return hex(num)