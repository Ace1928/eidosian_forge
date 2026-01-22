from .. import osutils
def decode_copy_instruction(bytes, cmd, pos):
    """Decode a copy instruction from the next few bytes.

    A copy instruction is a variable number of bytes, so we will parse the
    bytes we care about, and return the new position, as well as the offset and
    length referred to in the bytes.

    :param bytes: A string of bytes
    :param cmd: The command code
    :param pos: The position in bytes right after the copy command
    :return: (offset, length, newpos)
        The offset of the copy start, the number of bytes to copy, and the
        position after the last byte of the copy
    """
    if cmd & 128 != 128:
        raise ValueError('copy instructions must have bit 0x80 set')
    offset = 0
    length = 0
    if cmd & 1:
        offset = bytes[pos]
        pos += 1
    if cmd & 2:
        offset = offset | bytes[pos] << 8
        pos += 1
    if cmd & 4:
        offset = offset | bytes[pos] << 16
        pos += 1
    if cmd & 8:
        offset = offset | bytes[pos] << 24
        pos += 1
    if cmd & 16:
        length = bytes[pos]
        pos += 1
    if cmd & 32:
        length = length | bytes[pos] << 8
        pos += 1
    if cmd & 64:
        length = length | bytes[pos] << 16
        pos += 1
    if length == 0:
        length = 65536
    return (offset, length, pos)