from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def use_tsig(self, keyring, keyname=None, fudge=300, original_id=None, tsig_error=0, other_data=b'', algorithm=dns.tsig.default_algorithm):
    """When sending, a TSIG signature using the specified keyring
        and keyname should be added.

        See the documentation of the Message class for a complete
        description of the keyring dictionary.

        *keyring*, a ``dict``, the TSIG keyring to use.  If a
        *keyring* is specified but a *keyname* is not, then the key
        used will be the first key in the *keyring*.  Note that the
        order of keys in a dictionary is not defined, so applications
        should supply a keyname when a keyring is used, unless they
        know the keyring contains only one key.

        *keyname*, a ``dns.name.Name`` or ``None``, the name of the TSIG key
        to use; defaults to ``None``. The key must be defined in the keyring.

        *fudge*, an ``int``, the TSIG time fudge.

        *original_id*, an ``int``, the TSIG original id.  If ``None``,
        the message's id is used.

        *tsig_error*, an ``int``, the TSIG error code.

        *other_data*, a ``binary``, the TSIG other data.

        *algorithm*, a ``dns.name.Name``, the TSIG algorithm to use.
        """
    self.keyring = keyring
    if keyname is None:
        self.keyname = list(self.keyring.keys())[0]
    else:
        if isinstance(keyname, string_types):
            keyname = dns.name.from_text(keyname)
        self.keyname = keyname
    self.keyalgorithm = algorithm
    self.fudge = fudge
    if original_id is None:
        self.original_id = self.id
    else:
        self.original_id = original_id
    self.tsig_error = tsig_error
    self.other_data = other_data