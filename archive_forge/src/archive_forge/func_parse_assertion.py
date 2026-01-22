import calendar
import logging
from saml2 import SAMLError
from saml2 import class_name
from saml2 import extension_elements_to_elements
from saml2 import saml
from saml2 import samlp
from saml2 import time_util
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.attribute_converter import to_local
from saml2.s_utils import RequestVersionTooHigh
from saml2.s_utils import RequestVersionTooLow
from saml2.saml import SCM_BEARER
from saml2.saml import SCM_HOLDER_OF_KEY
from saml2.saml import SCM_SENDER_VOUCHES
from saml2.saml import XSI_TYPE
from saml2.saml import attribute_from_string
from saml2.saml import encrypted_attribute_from_string
from saml2.samlp import STATUS_AUTHN_FAILED
from saml2.samlp import STATUS_INVALID_ATTR_NAME_OR_VALUE
from saml2.samlp import STATUS_INVALID_NAMEID_POLICY
from saml2.samlp import STATUS_NO_AUTHN_CONTEXT
from saml2.samlp import STATUS_NO_AVAILABLE_IDP
from saml2.samlp import STATUS_NO_PASSIVE
from saml2.samlp import STATUS_NO_SUPPORTED_IDP
from saml2.samlp import STATUS_PARTIAL_LOGOUT
from saml2.samlp import STATUS_PROXY_COUNT_EXCEEDED
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_REQUEST_UNSUPPORTED
from saml2.samlp import STATUS_REQUEST_VERSION_DEPRECATED
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_HIGH
from saml2.samlp import STATUS_REQUEST_VERSION_TOO_LOW
from saml2.samlp import STATUS_RESOURCE_NOT_RECOGNIZED
from saml2.samlp import STATUS_RESPONDER
from saml2.samlp import STATUS_TOO_MANY_RESPONSES
from saml2.samlp import STATUS_UNKNOWN_ATTR_PROFILE
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.samlp import STATUS_UNSUPPORTED_BINDING
from saml2.samlp import STATUS_VERSION_MISMATCH
from saml2.sigver import DecryptError
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import signed
from saml2.time_util import later_than
from saml2.time_util import str_to_time
from saml2.validate import NotValid
from saml2.validate import valid_address
from saml2.validate import valid_instance
from saml2.validate import validate_before
from saml2.validate import validate_on_or_after
def parse_assertion(self, keys=None):
    """Parse the assertions for a saml response.

        :param keys: A string representing a RSA key or a list of strings
        containing RSA keys.
        :return: True if the assertions are parsed otherwise False.
        """
    if self.context == 'AuthnQuery':
        pass
    else:
        n_assertions = len(self.response.assertion)
        n_assertions_enc = len(self.response.encrypted_assertion)
        if n_assertions != 1 and n_assertions_enc != 1 and (self.assertion is None):
            raise InvalidAssertion(f'Invalid number of assertions in Response: {n_assertions + n_assertions_enc}')
    if self.response.assertion:
        logger.debug('***Unencrypted assertion***')
        for assertion in self.response.assertion:
            if not self._assertion(assertion, False):
                return False
    if self.find_encrypt_data(self.response):
        logger.debug('***Encrypted assertion/-s***')
        _enc_assertions = []
        resp = self.response
        decr_text = str(self.response)
        decr_text_old = None
        while self.find_encrypt_data(resp) and decr_text_old != decr_text:
            decr_text_old = decr_text
            try:
                decr_text = self.sec.decrypt_keys(decr_text, keys=keys)
            except DecryptError:
                continue
            else:
                resp = samlp.response_from_string(decr_text)
                if type(decr_text_old) != type(decr_text):
                    if isinstance(decr_text_old, bytes):
                        decr_text_old = decr_text_old.decode('utf-8')
                    else:
                        decr_text_old = decr_text_old.encode('utf-8')
        _enc_assertions = self.decrypt_assertions(resp.encrypted_assertion, decr_text)
        decr_text_old = None
        while (self.find_encrypt_data(resp) or self.find_encrypt_data_assertion_list(_enc_assertions)) and decr_text_old != decr_text:
            decr_text_old = decr_text
            try:
                decr_text = self.sec.decrypt_keys(decr_text, keys=keys)
            except DecryptError:
                continue
            else:
                resp = samlp.response_from_string(decr_text)
                _enc_assertions = self.decrypt_assertions(resp.encrypted_assertion, decr_text, verified=True)
                if type(decr_text_old) != type(decr_text):
                    if isinstance(decr_text_old, bytes):
                        decr_text_old = decr_text_old.decode('utf-8')
                    else:
                        decr_text_old = decr_text_old.encode('utf-8')
        all_assertions = _enc_assertions
        if resp.assertion:
            all_assertions = all_assertions + resp.assertion
        if len(all_assertions) > 0:
            for tmp_ass in all_assertions:
                if tmp_ass.advice and tmp_ass.advice.encrypted_assertion:
                    advice_res = self.decrypt_assertions(tmp_ass.advice.encrypted_assertion, decr_text, tmp_ass.issuer)
                    if tmp_ass.advice.assertion:
                        tmp_ass.advice.assertion.extend(advice_res)
                    else:
                        tmp_ass.advice.assertion = advice_res
                    if len(advice_res) > 0:
                        tmp_ass.advice.encrypted_assertion = []
        self.response.assertion = resp.assertion
        for assertion in _enc_assertions:
            if not self._assertion(assertion, True):
                return False
            else:
                self.assertions.append(assertion)
        self.xmlstr = decr_text
        if len(_enc_assertions) > 0:
            self.response.encrypted_assertion = []
    if self.response.assertion:
        for assertion in self.response.assertion:
            self.assertions.append(assertion)
    if self.assertions and len(self.assertions) > 0:
        self.assertion = self.assertions[0]
    if self.context == 'AuthnReq' or self.context == 'AttrQuery':
        self.ava = self.get_identity()
        logger.debug(f'--- AVA: {self.ava}')
    return True