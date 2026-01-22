import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def parse_soap_enveloped_saml_thingy(text, expected_tags):
    """Parses a SOAP enveloped SAML thing and returns the thing as
    a string.

    :param text: The SOAP object as XML string
    :param expected_tags: What the tag of the SAML thingy is expected to be.
    :return: SAML thingy as a string
    """
    envelope = defusedxml.ElementTree.fromstring(text)
    envelope_tag = '{%s}Envelope' % soapenv.NAMESPACE
    if envelope.tag != envelope_tag:
        raise ValueError(f"Invalid envelope tag '{envelope.tag}' should be '{envelope_tag}'")
    if len(envelope) < 1:
        raise Exception('No items in envelope.')
    body = None
    for part in envelope:
        if part.tag == '{%s}Body' % soapenv.NAMESPACE:
            n_children = len(part)
            if n_children != 1:
                raise Exception(f'Expected a single child element, found {n_children}')
            body = part
            break
    if body is None:
        return ''
    saml_part = body[0]
    if saml_part.tag in expected_tags:
        return ElementTree.tostring(saml_part, encoding='UTF-8')
    else:
        raise WrongMessageType(f"Was '{saml_part.tag}' expected one of {expected_tags}")