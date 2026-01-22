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
def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
    if asn1Spec is None:
        isTagged = False
    elif asn1Spec.__class__ is tagmap.TagMap:
        isTagged = tagSet in asn1Spec.tagMap
    else:
        isTagged = tagSet == asn1Spec.tagSet
    if isTagged:
        chunk = null
        if LOG:
            LOG('decoding as tagged ANY')
    else:
        fullPosition = substrate.markedPosition
        currentPosition = substrate.tell()
        substrate.seek(fullPosition, os.SEEK_SET)
        for chunk in readFromStream(substrate, currentPosition - fullPosition, options):
            if isinstance(chunk, SubstrateUnderrunError):
                yield chunk
        if LOG:
            LOG('decoding as untagged ANY, header substrate %s' % debug.hexdump(chunk))
    asn1Spec = self.protoComponent
    if substrateFun and substrateFun is not self.substrateCollector:
        asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
        for chunk in substrateFun(asn1Object, chunk + substrate, length + len(chunk), options):
            yield chunk
        return
    if LOG:
        LOG('assembling constructed serialization')
    substrateFun = self.substrateCollector
    while True:
        for component in decodeFun(substrate, asn1Spec, substrateFun=substrateFun, allowEoo=True, **options):
            if isinstance(component, SubstrateUnderrunError):
                yield component
            if component is eoo.endOfOctets:
                break
        if component is eoo.endOfOctets:
            break
        chunk += component
    if substrateFun:
        yield chunk
    else:
        yield self._createComponent(asn1Spec, tagSet, chunk, **options)