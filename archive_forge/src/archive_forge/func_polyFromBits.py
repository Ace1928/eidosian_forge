from crcmod import Crc
def polyFromBits(bits):
    p = 0
    for n in bits:
        p = p | 1 << n
    return p