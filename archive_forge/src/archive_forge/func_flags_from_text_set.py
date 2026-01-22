import base64
import struct
import dns.exception
import dns.dnssec
import dns.rdata
def flags_from_text_set(texts_set):
    """Convert set of DNSKEY flag mnemonic texts to DNSKEY flag value
    @rtype: int"""
    flags = 0
    for text in texts_set:
        try:
            flags += _flag_by_text[text]
        except KeyError:
            raise NotImplementedError("DNSKEY flag '%s' is not supported" % text)
    return flags