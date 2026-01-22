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
def http_redirect_message(message, location, relay_state='', typ='SAMLRequest', sigalg=None, sign=None, backend=None):
    """The HTTP Redirect binding defines a mechanism by which SAML protocol
    messages can be transmitted within URL parameters.
    Messages are encoded for use with this binding using a URL encoding
    technique, and transmitted using the HTTP GET method.

    The DEFLATE Encoding is used in this function.

    :param message: The message
    :param location: Where the message should be posted to
    :param relay_state: for preserving and conveying state information
    :param typ: What type of message it is SAMLRequest/SAMLResponse/SAMLart
    :param sigalg: Which algorithm the signature function will use to sign
        the message
    :param sign: Whether the message should be signed
    :return: A tuple containing header information and a HTML message.
    """
    if not isinstance(message, str):
        message = f'{message}'
    _order = None
    if typ in ['SAMLRequest', 'SAMLResponse']:
        if typ == 'SAMLRequest':
            _order = REQ_ORDER
        else:
            _order = RESP_ORDER
        args = {typ: deflate_and_base64_encode(message)}
    elif typ == 'SAMLart':
        args = {typ: message}
    else:
        raise Exception(f'Unknown message type: {typ}')
    if relay_state:
        args['RelayState'] = relay_state
    if sign:
        if sigalg not in [long_name for short_name, long_name in SIG_ALLOWED_ALG]:
            raise Exception(f'Signature algo not in allowed list: {sigalg}')
        signer = backend.get_signer(sigalg) if sign and sigalg else None
        if not signer:
            raise Exception(f'Could not init signer fro algo {sigalg}')
        args['SigAlg'] = sigalg
        string = '&'.join((urlencode({k: args[k]}) for k in _order if k in args))
        string_enc = string.encode('ascii')
        args['Signature'] = base64.b64encode(signer.sign(string_enc))
    string = urlencode(args)
    glue_char = '&' if urlparse(location).query else '?'
    login_url = glue_char.join([location, string])
    headers = [('Location', str(login_url))]
    body = []
    return {'headers': headers, 'data': body, 'status': 303}