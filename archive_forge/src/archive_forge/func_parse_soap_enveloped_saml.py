import base64
import logging
from urllib.parse import urlencode
from urllib.parse import urlparse
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
import saml2
from saml2.s_utils import deflate_and_base64_encode
from saml2.sigver import REQ_ORDER
from saml2.sigver import RESP_ORDER
from saml2.xmldsig import SIG_ALLOWED_ALG
def parse_soap_enveloped_saml(text, body_class, header_class=None):
    """Parses a SOAP enveloped SAML thing and returns header parts and body

    :param text: The SOAP object as XML
    :return: header parts and body as saml.samlbase instances
    """
    envelope = defusedxml.ElementTree.fromstring(text)
    envelope_tag = '{%s}Envelope' % NAMESPACE
    if envelope.tag != envelope_tag:
        raise ValueError(f"Invalid envelope tag '{envelope.tag}' should be '{envelope_tag}'")
    body = None
    header = {}
    for part in envelope:
        if part.tag == '{%s}Body' % NAMESPACE:
            for sub in part:
                try:
                    body = saml2.create_class_from_element_tree(body_class, sub)
                except Exception:
                    raise Exception(f'Wrong body type ({sub.tag}) in SOAP envelope')
        elif part.tag == '{%s}Header' % NAMESPACE:
            if not header_class:
                raise Exception("Header where I didn't expect one")
            for sub in part:
                for klass in header_class:
                    if sub.tag == f'{{{klass.c_namespace}}}{klass.c_tag}':
                        header[sub.tag] = saml2.create_class_from_element_tree(klass, sub)
                        break
    return (body, header)