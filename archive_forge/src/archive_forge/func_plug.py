import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
def plug(self, *args, **kw):
    raise error.PyAsn1Error('Attempted "%s" operation on ASN.1 schema object' % name)