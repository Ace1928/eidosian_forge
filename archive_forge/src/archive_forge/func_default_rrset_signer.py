import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
def default_rrset_signer(txn: dns.transaction.Transaction, rrset: dns.rrset.RRset, signer: dns.name.Name, ksks: List[Tuple[PrivateKey, DNSKEY]], zsks: List[Tuple[PrivateKey, DNSKEY]], inception: Optional[Union[datetime, str, int, float]]=None, expiration: Optional[Union[datetime, str, int, float]]=None, lifetime: Optional[int]=None, policy: Optional[Policy]=None, origin: Optional[dns.name.Name]=None) -> None:
    """Default RRset signer"""
    if rrset.rdtype in set([dns.rdatatype.RdataType.DNSKEY, dns.rdatatype.RdataType.CDS, dns.rdatatype.RdataType.CDNSKEY]):
        keys = ksks
    else:
        keys = zsks
    for private_key, dnskey in keys:
        rrsig = dns.dnssec.sign(rrset=rrset, private_key=private_key, dnskey=dnskey, inception=inception, expiration=expiration, lifetime=lifetime, signer=signer, policy=policy, origin=origin)
        txn.add(rrset.name, rrset.ttl, rrsig)