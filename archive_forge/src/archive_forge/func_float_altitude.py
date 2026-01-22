import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer
@property
def float_altitude(self):
    """altitude as a floating point value"""
    return float(self.altitude)