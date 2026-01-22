import struct, warnings
def tag2num(n):
    try:
        return int(n)
    except ValueError:
        n = (n + '    ')[:4]
        return struct.unpack('>L', n.encode('ascii'))[0]