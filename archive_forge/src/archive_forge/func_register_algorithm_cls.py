from typing import Dict, Optional, Tuple, Type, Union
import dns.name
from dns.dnssecalgs.base import GenericPrivateKey
from dns.dnssectypes import Algorithm
from dns.exception import UnsupportedAlgorithm
from dns.rdtypes.ANY.DNSKEY import DNSKEY
def register_algorithm_cls(algorithm: Union[int, str], algorithm_cls: Type[GenericPrivateKey], name: Optional[Union[dns.name.Name, str]]=None, oid: Optional[bytes]=None) -> None:
    """Register Algorithm Private Key class.

    *algorithm*, a ``str`` or ``int`` specifying the DNSKEY algorithm.

    *algorithm_cls*: A `GenericPrivateKey` class.

    *name*, an optional ``dns.name.Name`` or ``str``, for for PRIVATEDNS algorithms.

    *oid*: an optional BER-encoded `bytes` for PRIVATEOID algorithms.

    Raises ``ValueError`` if a name or oid is specified incorrectly.
    """
    if not issubclass(algorithm_cls, GenericPrivateKey):
        raise TypeError('Invalid algorithm class')
    algorithm = Algorithm.make(algorithm)
    prefix: AlgorithmPrefix = None
    if algorithm == Algorithm.PRIVATEDNS:
        if name is None:
            raise ValueError('Name required for PRIVATEDNS algorithms')
        if isinstance(name, str):
            name = dns.name.from_text(name)
        prefix = name
    elif algorithm == Algorithm.PRIVATEOID:
        if oid is None:
            raise ValueError('OID required for PRIVATEOID algorithms')
        prefix = bytes([len(oid)]) + oid
    elif name:
        raise ValueError('Name only supported for PRIVATEDNS algorithm')
    elif oid:
        raise ValueError('OID only supported for PRIVATEOID algorithm')
    algorithms[algorithm, prefix] = algorithm_cls