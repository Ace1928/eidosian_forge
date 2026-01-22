import struct, warnings
def bininfo(num, size=1):
    if num == 0:
        return struct.pack('>4H', 0, 0, 0, 0)
    srange = 1
    select = 0
    while srange <= num:
        srange *= 2
        select += 1
    select -= 1
    srange //= 2
    srange *= size
    shift = num * size - srange
    return struct.pack('>4H', num, srange, select, shift)