import calendar
import copy
import http.cookiejar as http_cookiejar
from http.cookies import SimpleCookie
import logging
import re
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
import requests
from saml2 import SAMLError
from saml2 import class_name
from saml2.pack import make_soap_enveloped_saml_thingy
from saml2.time_util import utc_now
def use_soap(self, request, destination='', soap_headers=None, sign=False, **kwargs):
    """
        Construct the necessary information for using SOAP+POST

        :param request:
        :param destination:
        :param soap_headers:
        :param sign:
        :return: dictionary
        """
    headers = [('content-type', 'application/soap+xml')]
    soap_message = make_soap_enveloped_saml_thingy(request, soap_headers)
    logger.debug('SOAP message: %s', soap_message)
    if sign and self.sec:
        _signed = self.sec.sign_statement(soap_message, node_name=class_name(request), node_id=request.id)
        soap_message = _signed
    return {'url': destination, 'method': 'POST', 'data': soap_message, 'headers': headers}