import struct
import dns.exception
import dns.immutable
import dns.rdata
@property
def float_latitude(self):
    """latitude as a floating point value"""
    return _tuple_to_float(self.latitude)