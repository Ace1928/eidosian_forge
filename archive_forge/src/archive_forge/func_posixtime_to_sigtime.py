import base64
import calendar
import struct
import time
import dns.dnssec
import dns.exception
import dns.rdata
import dns.rdatatype
def posixtime_to_sigtime(what):
    return time.strftime('%Y%m%d%H%M%S', time.gmtime(what))