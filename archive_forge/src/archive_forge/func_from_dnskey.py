import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dsa, utils
from dns.dnssecalgs.cryptography import CryptographyPrivateKey, CryptographyPublicKey
from dns.dnssectypes import Algorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
@classmethod
def from_dnskey(cls, key: DNSKEY) -> 'PublicDSA':
    cls._ensure_algorithm_key_combination(key)
    keyptr = key.key
    t, = struct.unpack('!B', keyptr[0:1])
    keyptr = keyptr[1:]
    octets = 64 + t * 8
    dsa_q = keyptr[0:20]
    keyptr = keyptr[20:]
    dsa_p = keyptr[0:octets]
    keyptr = keyptr[octets:]
    dsa_g = keyptr[0:octets]
    keyptr = keyptr[octets:]
    dsa_y = keyptr[0:octets]
    return cls(key=dsa.DSAPublicNumbers(int.from_bytes(dsa_y, 'big'), dsa.DSAParameterNumbers(int.from_bytes(dsa_p, 'big'), int.from_bytes(dsa_q, 'big'), int.from_bytes(dsa_g, 'big'))).public_key(default_backend()))