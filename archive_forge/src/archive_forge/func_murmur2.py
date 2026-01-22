import random
def murmur2(data):
    """Pure-python Murmur2 implementation.

    Based on java client, see org.apache.kafka.common.utils.Utils.murmur2

    Args:
        data (bytes): opaque bytes

    Returns: MurmurHash2 of data
    """
    length = len(data)
    seed = 2538058380
    m = 1540483477
    r = 24
    h = seed ^ length
    length4 = length // 4
    for i in range(length4):
        i4 = i * 4
        k = (data[i4 + 0] & 255) + ((data[i4 + 1] & 255) << 8) + ((data[i4 + 2] & 255) << 16) + ((data[i4 + 3] & 255) << 24)
        k &= 4294967295
        k *= m
        k &= 4294967295
        k ^= k % 4294967296 >> r
        k &= 4294967295
        k *= m
        k &= 4294967295
        h *= m
        h &= 4294967295
        h ^= k
        h &= 4294967295
    extra_bytes = length % 4
    if extra_bytes >= 3:
        h ^= (data[(length & ~3) + 2] & 255) << 16
        h &= 4294967295
    if extra_bytes >= 2:
        h ^= (data[(length & ~3) + 1] & 255) << 8
        h &= 4294967295
    if extra_bytes >= 1:
        h ^= data[length & ~3] & 255
        h &= 4294967295
        h *= m
        h &= 4294967295
    h ^= h % 4294967296 >> 13
    h &= 4294967295
    h *= m
    h &= 4294967295
    h ^= h % 4294967296 >> 15
    h &= 4294967295
    return h