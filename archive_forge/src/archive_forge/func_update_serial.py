import collections
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl
def update_serial(self, value: int=1, relative: bool=True, name: dns.name.Name=dns.name.empty) -> None:
    """Update the serial number.

        *value*, an `int`, is an increment if *relative* is `True`, or the
        actual value to set if *relative* is `False`.

        Raises `KeyError` if there is no SOA rdataset at *name*.

        Raises `ValueError` if *value* is negative or if the increment is
        so large that it would cause the new serial to be less than the
        prior value.
        """
    self._check_ended()
    if value < 0:
        raise ValueError('negative update_serial() value')
    if isinstance(name, str):
        name = dns.name.from_text(name, None)
    rdataset = self._get_rdataset(name, dns.rdatatype.SOA, dns.rdatatype.NONE)
    if rdataset is None or len(rdataset) == 0:
        raise KeyError
    if relative:
        serial = dns.serial.Serial(rdataset[0].serial) + value
    else:
        serial = dns.serial.Serial(value)
    serial = serial.value
    if serial == 0:
        serial = 1
    rdata = rdataset[0].replace(serial=serial)
    new_rdataset = dns.rdataset.from_rdata(rdataset.ttl, rdata)
    self.replace(name, new_rdataset)