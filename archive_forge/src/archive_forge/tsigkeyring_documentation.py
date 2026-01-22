from dns._compat import maybe_decode, maybe_encode
import base64
import dns.name
Convert a dictionary containing (dns.name.Name, binary secret) pairs
    into a text keyring which has (textual DNS name, base64 secret) pairs.
    @rtype: dict