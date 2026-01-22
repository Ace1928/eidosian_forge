from abc import ABC, abstractmethod  # pylint: disable=no-name-in-module
from typing import Any, Optional, Type
import dns.rdataclass
import dns.rdatatype
from dns.dnssectypes import Algorithm
from dns.exception import AlgorithmKeyMismatch
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.dnskeybase import Flag
def to_dnskey(self, flags: int=Flag.ZONE, protocol: int=3) -> DNSKEY:
    """Return public key as DNSKEY"""
    return DNSKEY(rdclass=dns.rdataclass.IN, rdtype=dns.rdatatype.DNSKEY, flags=flags, protocol=protocol, algorithm=self.algorithm, key=self.encode_key_bytes())