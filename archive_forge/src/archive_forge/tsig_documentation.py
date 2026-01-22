import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
Return the tsig algorithm for the specified tsig_rdata
    @raises FormError: The TSIG is badly formed.
    