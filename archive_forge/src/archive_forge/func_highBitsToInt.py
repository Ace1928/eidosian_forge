@staticmethod
def highBitsToInt(value):
    bit = ord(value) if type(value) is str else value
    return (bit & 255) >> 4