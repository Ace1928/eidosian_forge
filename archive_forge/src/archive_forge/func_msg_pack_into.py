import struct
def msg_pack_into(fmt, buf, offset, *args):
    needed_len = offset + struct.calcsize(fmt)
    if len(buf) < needed_len:
        buf += bytearray(needed_len - len(buf))
    struct.pack_into(fmt, buf, offset, *args)