from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def nextprefix(self):
    """
        Get the next available prefix.  This means a prefix starting with 'ns' with
        a number appended as (ns0, ns1, ..) that is not already defined in the
        WSDL document.
        """
    used = [ns[0] for ns in self.prefixes]
    used += [ns[0] for ns in list(self.wsdl.root.nsprefixes.items())]
    for n in range(0, 1024):
        p = 'ns%d' % n
        if p not in used:
            return p
    raise Exception('prefixes exhausted')