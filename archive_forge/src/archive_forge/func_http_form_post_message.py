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
def http_form_post_message(message, location, relay_state='', typ='SAMLRequest', **kwargs):
    """The HTTP POST binding defines a mechanism by which SAML protocol
    messages may be transmitted within the base64-encoded content of a
    HTML form control.

    :param message: The message
    :param location: Where the form should be posted to
    :param relay_state: for preserving and conveying state information
    :return: A tuple containing header information and a HTML message.
    """
    if not isinstance(message, str):
        message = str(message)
    if not isinstance(message, bytes):
        message = message.encode('utf-8')
    if typ == 'SAMLRequest' or typ == 'SAMLResponse':
        _msg = base64.b64encode(message)
    else:
        _msg = message
    _msg = _msg.decode('ascii')
    saml_response_input = HTML_INPUT_ELEMENT_SPEC.format(name=_html_escape(typ), val=_html_escape(_msg), type='hidden')
    relay_state_input = ''
    if relay_state:
        relay_state_input = HTML_INPUT_ELEMENT_SPEC.format(name='RelayState', val=_html_escape(relay_state), type='hidden')
    response = HTML_FORM_SPEC.format(saml_response_input=saml_response_input, relay_state_input=relay_state_input, action=location)
    return {'headers': [('Content-type', 'text/html')], 'data': response, 'status': 200}