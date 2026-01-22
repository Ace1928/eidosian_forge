import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
@property
def tagMap(self):
    """"Return a :class:`~pyasn1.type.tagmap.TagMap` object mapping
            ASN.1 tags to ASN.1 objects contained within callee.
        """
    try:
        return self._tagMap
    except AttributeError:
        self._tagMap = tagmap.TagMap({self.tagSet: self}, {eoo.endOfOctets.tagSet: eoo.endOfOctets}, self)
        return self._tagMap