from sys import version_info
def bin(value):
    bitstring = []
    if value > 0:
        prefix = '0b'
    elif value < 0:
        prefix = '-0b'
        value = abs(value)
    else:
        prefix = '0b0'
    while value:
        if value & 1 == 1:
            bitstring.append('1')
        else:
            bitstring.append('0')
        value >>= 1
    bitstring.reverse()
    return prefix + ''.join(bitstring)