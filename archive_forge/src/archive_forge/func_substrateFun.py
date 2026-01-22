import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
def substrateFun(asn1Object, _substrate, _length, _options):
    """Legacy hack to keep the recursiveFlag=False option supported.

                        The decode(..., substrateFun=userCallback) option was introduced in 0.1.4 as a generalization
                        of the old recursiveFlag=False option. Users should pass their callback instead of using
                        recursiveFlag.
                        """
    yield asn1Object