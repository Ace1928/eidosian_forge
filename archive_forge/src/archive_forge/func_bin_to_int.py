from .py3compat import int2byte
def bin_to_int(bits, signed=False):
    """
    Logical opposite of int_to_bin. Both '0' and '\\x00' are considered zero,
    and both '1' and '\\x01' are considered one. Set sign to True to interpret
    the number as a 2-s complement signed integer.
    """
    number = 0
    bias = 0
    ptr = 0
    if signed and _bit_values[bits[0]] == 1:
        bits = bits[1:]
        bias = 1 << len(bits)
    for b in bits:
        number <<= 1
        number |= _bit_values[b]
    return number - bias