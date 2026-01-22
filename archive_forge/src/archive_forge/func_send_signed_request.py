from __future__ import absolute_import, division, print_function
import copy
import datetime
import json
import locale
import time
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import PY3
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_openssl_cli import (
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def send_signed_request(self, url, payload, key_data=None, jws_header=None, parse_json_result=True, encode_payload=True, fail_on_error=True, error_msg=None, expected_status_codes=None):
    """
        Sends a JWS signed HTTP POST request to the ACME server and returns
        the response as dictionary (if parse_json_result is True) or in raw form
        (if parse_json_result is False).
        https://tools.ietf.org/html/rfc8555#section-6.2

        If payload is None, a POST-as-GET is performed.
        (https://tools.ietf.org/html/rfc8555#section-6.3)
        """
    key_data = key_data or self.account_key_data
    jws_header = jws_header or self.account_jws_header
    failed_tries = 0
    while True:
        protected = copy.deepcopy(jws_header)
        protected['nonce'] = self.directory.get_nonce()
        if self.version != 1:
            protected['url'] = url
        self._log('URL', url)
        self._log('protected', protected)
        self._log('payload', payload)
        data = self.sign_request(protected, payload, key_data, encode_payload=encode_payload)
        if self.version == 1:
            data['header'] = jws_header.copy()
            for k, v in protected.items():
                dummy = data['header'].pop(k, None)
        self._log('signed request', data)
        data = self.module.jsonify(data)
        headers = {'Content-Type': 'application/jose+json'}
        resp, info = fetch_url(self.module, url, data=data, headers=headers, method='POST', timeout=self.request_timeout)
        if _decode_retry(self.module, resp, info, failed_tries):
            failed_tries += 1
            continue
        _assert_fetch_url_success(self.module, resp, info)
        result = {}
        try:
            if PY3 and resp.closed:
                raise TypeError
            content = resp.read()
        except (AttributeError, TypeError):
            content = info.pop('body', None)
        if content or not parse_json_result:
            if parse_json_result and info['content-type'].startswith('application/json') or 400 <= info['status'] < 600:
                try:
                    decoded_result = self.module.from_json(content.decode('utf8'))
                    self._log('parsed result', decoded_result)
                    if all((400 <= info['status'] < 600, decoded_result.get('type') == 'urn:ietf:params:acme:error:badNonce', failed_tries <= 5)):
                        failed_tries += 1
                        continue
                    if parse_json_result:
                        result = decoded_result
                    else:
                        result = content
                except ValueError:
                    raise NetworkException('Failed to parse the ACME response: {0} {1}'.format(url, content))
            else:
                result = content
        if fail_on_error and _is_failed(info, expected_status_codes=expected_status_codes):
            raise ACMEProtocolException(self.module, msg=error_msg, info=info, content=content, content_json=result if parse_json_result else None)
        return (result, info)