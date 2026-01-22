import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def do_attribute_statement(identity):
    """
    :param identity: A dictionary with fiendly names as keys
    :return:
    """
    return saml.AttributeStatement(attribute=do_attributes(identity))