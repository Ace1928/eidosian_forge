import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def decode_integer(data, prefix_bits):
    """
    This decodes an integer according to the wacky integer encoding rules
    defined in the HPACK spec. Returns a tuple of the decoded integer and the
    number of bytes that were consumed from ``data`` in order to get that
    integer.
    """
    if prefix_bits < 1 or prefix_bits > 8:
        raise ValueError('Prefix bits must be between 1 and 8, got %s' % prefix_bits)
    max_number = _PREFIX_BIT_MAX_NUMBERS[prefix_bits]
    index = 1
    shift = 0
    mask = 255 >> 8 - prefix_bits
    try:
        number = to_byte(data[0]) & mask
        if number == max_number:
            while True:
                next_byte = to_byte(data[index])
                index += 1
                if next_byte >= 128:
                    number += next_byte - 128 << shift
                else:
                    number += next_byte << shift
                    break
                shift += 7
    except IndexError:
        raise HPACKDecodingError('Unable to decode HPACK integer representation from %r' % data)
    log.debug('Decoded %d, consumed %d bytes', number, index)
    return (number, index)