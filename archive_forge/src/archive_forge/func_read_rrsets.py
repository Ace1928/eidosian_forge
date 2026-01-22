import re
import sys
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
import dns.exception
import dns.grange
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.transaction
import dns.ttl
def read_rrsets(text: Any, name: Optional[Union[dns.name.Name, str]]=None, ttl: Optional[int]=None, rdclass: Optional[Union[dns.rdataclass.RdataClass, str]]=dns.rdataclass.IN, default_rdclass: Union[dns.rdataclass.RdataClass, str]=dns.rdataclass.IN, rdtype: Optional[Union[dns.rdatatype.RdataType, str]]=None, default_ttl: Optional[Union[int, str]]=None, idna_codec: Optional[dns.name.IDNACodec]=None, origin: Optional[Union[dns.name.Name, str]]=dns.name.root, relativize: bool=False) -> List[dns.rrset.RRset]:
    """Read one or more rrsets from the specified text, possibly subject
    to restrictions.

    *text*, a file object or a string, is the input to process.

    *name*, a string, ``dns.name.Name``, or ``None``, is the owner name of
    the rrset.  If not ``None``, then the owner name is "forced", and the
    input must not specify an owner name.  If ``None``, then any owner names
    are allowed and must be present in the input.

    *ttl*, an ``int``, string, or None.  If not ``None``, the the TTL is
    forced to be the specified value and the input must not specify a TTL.
    If ``None``, then a TTL may be specified in the input.  If it is not
    specified, then the *default_ttl* will be used.

    *rdclass*, a ``dns.rdataclass.RdataClass``, string, or ``None``.  If
    not ``None``, then the class is forced to the specified value, and the
    input must not specify a class.  If ``None``, then the input may specify
    a class that matches *default_rdclass*.  Note that it is not possible to
    return rrsets with differing classes; specifying ``None`` for the class
    simply allows the user to optionally type a class as that may be convenient
    when cutting and pasting.

    *default_rdclass*, a ``dns.rdataclass.RdataClass`` or string.  The class
    of the returned rrsets.

    *rdtype*, a ``dns.rdatatype.RdataType``, string, or ``None``.  If not
    ``None``, then the type is forced to the specified value, and the
    input must not specify a type.  If ``None``, then a type must be present
    for each RR.

    *default_ttl*, an ``int``, string, or ``None``.  If not ``None``, then if
    the TTL is not forced and is not specified, then this value will be used.
    if ``None``, then if the TTL is not forced an error will occur if the TTL
    is not specified.

    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA
    encoder/decoder.  If ``None``, the default IDNA 2003 encoder/decoder
    is used.  Note that codecs only apply to the owner name; dnspython does
    not do IDNA for names in rdata, as there is no IDNA zonefile format.

    *origin*, a string, ``dns.name.Name``, or ``None``, is the origin for any
    relative names in the input, and also the origin to relativize to if
    *relativize* is ``True``.

    *relativize*, a bool.  If ``True``, names are relativized to the *origin*;
    if ``False`` then any relative names in the input are made absolute by
    appending the *origin*.
    """
    if isinstance(origin, str):
        origin = dns.name.from_text(origin, dns.name.root, idna_codec)
    if isinstance(name, str):
        name = dns.name.from_text(name, origin, idna_codec)
    if isinstance(ttl, str):
        ttl = dns.ttl.from_text(ttl)
    if isinstance(default_ttl, str):
        default_ttl = dns.ttl.from_text(default_ttl)
    if rdclass is not None:
        rdclass = dns.rdataclass.RdataClass.make(rdclass)
    else:
        rdclass = None
    default_rdclass = dns.rdataclass.RdataClass.make(default_rdclass)
    if rdtype is not None:
        rdtype = dns.rdatatype.RdataType.make(rdtype)
    else:
        rdtype = None
    manager = RRSetsReaderManager(origin, relativize, default_rdclass)
    with manager.writer(True) as txn:
        tok = dns.tokenizer.Tokenizer(text, '<input>', idna_codec=idna_codec)
        reader = Reader(tok, default_rdclass, txn, allow_directives=False, force_name=name, force_ttl=ttl, force_rdclass=rdclass, force_rdtype=rdtype, default_ttl=default_ttl)
        reader.read()
    return manager.rrsets