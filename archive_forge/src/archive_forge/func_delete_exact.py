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
def delete_exact(self, *args: Any) -> None:
    """Delete records.

        The arguments may be:

            - rrset

            - name

            - name, rdatatype, [covers]

            - name, rdataset...

            - name, rdata...

        Raises dns.transaction.DeleteNotExact if some of the records
        are not in the existing set.

        """
    self._check_ended()
    self._check_read_only()
    self._delete(True, args)