import sys
def get_word_alignment(num, force_arch=64, _machine_word_size=MACHINE_WORD_SIZE):
    """
    Returns alignment details for the given number based on the platform
    Python is running on.

    :param num:
        Unsigned integral number.
    :param force_arch:
        If you don't want to use 64-bit unsigned chunks, set this to
        anything other than 64. 32-bit chunks will be preferred then.
        Default 64 will be used when on a 64-bit machine.
    :param _machine_word_size:
        (Internal) The machine word size used for alignment.
    :returns:
        4-tuple::

            (word_bits, word_bytes,
             max_uint, packing_format_type)
    """
    max_uint64 = 18446744073709551615
    max_uint32 = 4294967295
    max_uint16 = 65535
    max_uint8 = 255
    if force_arch == 64 and _machine_word_size >= 64 and (num > max_uint32):
        return (64, 8, max_uint64, 'Q')
    elif num > max_uint16:
        return (32, 4, max_uint32, 'L')
    elif num > max_uint8:
        return (16, 2, max_uint16, 'H')
    else:
        return (8, 1, max_uint8, 'B')