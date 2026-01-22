import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def xml_qname_to_QualifiedName(element, qname_str):
    if ':' in qname_str:
        prefix, localpart = qname_str.split(':', 1)
        if prefix in element.nsmap:
            ns_uri = element.nsmap[prefix]
            if ns_uri == XML_XSD_URI:
                ns = XSD
            elif ns_uri == PROV.uri:
                ns = PROV
            else:
                ns = Namespace(prefix, ns_uri)
            return ns[localpart]
    if None in element.nsmap:
        ns_uri = element.nsmap[None]
        ns = Namespace('', ns_uri)
        return ns[qname_str]
    raise ProvXMLException('Could not create a valid QualifiedName for "%s"' % qname_str)