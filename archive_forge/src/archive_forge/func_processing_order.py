import io
import random
import struct
from typing import Any, Collection, Dict, List, Optional, Union, cast
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.renderer
import dns.set
import dns.ttl
def processing_order(self) -> List[dns.rdata.Rdata]:
    """Return rdatas in a valid processing order according to the type's
        specification.  For example, MX records are in preference order from
        lowest to highest preferences, with items of the same preference
        shuffled.

        For types that do not define a processing order, the rdatas are
        simply shuffled.
        """
    if len(self) == 0:
        return []
    else:
        return self[0]._processing_order(iter(self))