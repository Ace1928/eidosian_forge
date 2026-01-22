import contextlib
import io
import os
import struct
from typing import (
import dns.exception
import dns.grange
import dns.immutable
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rdtypes.ANY.ZONEMD
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
import dns.zonefile
from dns.zonetypes import DigestHashAlgorithm, DigestScheme, _digest_hashers
def get_soa(self, txn: Optional[dns.transaction.Transaction]=None) -> dns.rdtypes.ANY.SOA.SOA:
    """Get the zone SOA rdata.

        Raises ``dns.zone.NoSOA`` if there is no SOA RRset.

        Returns a ``dns.rdtypes.ANY.SOA.SOA`` Rdata.
        """
    if self.relativize:
        origin_name = dns.name.empty
    else:
        if self.origin is None:
            raise NoSOA
        origin_name = self.origin
    soa: Optional[dns.rdataset.Rdataset]
    if txn:
        soa = txn.get(origin_name, dns.rdatatype.SOA)
    else:
        soa = self.get_rdataset(origin_name, dns.rdatatype.SOA)
    if soa is None:
        raise NoSOA
    return soa[0]