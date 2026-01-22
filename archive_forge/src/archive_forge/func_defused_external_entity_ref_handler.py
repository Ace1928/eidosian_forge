from __future__ import print_function, absolute_import
from xml.sax.expatreader import ExpatParser as _ExpatParser
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
def defused_external_entity_ref_handler(self, context, base, sysid, pubid):
    raise ExternalReferenceForbidden(context, base, sysid, pubid)