from .py3compat import int2byte
def decode_bin(data):
    """
    Locical opposite of decode_bin.
    """
    if len(data) & 7:
        raise ValueError('Data length must be a multiple of 8')
    i = 0
    j = 0
    l = len(data) // 8
    chars = [b''] * l
    while j < l:
        chars[j] = _bin_to_char[data[i:i + 8]]
        i += 8
        j += 1
    return b''.join(chars)