from __future__ import unicode_literals
import sys
import os
import requests
import requests.auth
import warnings
from winrm.exceptions import InvalidCredentialsError, WinRMError, WinRMTransportError
from winrm.encryption import Encryption
def build_session(self):
    session = requests.Session()
    proxies = dict()
    if self.proxy is None:
        proxies['no_proxy'] = '*'
    elif self.proxy != 'legacy_requests':
        proxies['http'] = self.proxy
        proxies['https'] = self.proxy
    settings = session.merge_environment_settings(url=self.endpoint, proxies=proxies, stream=None, verify=None, cert=None)
    global DISPLAYED_PROXY_WARNING
    if not DISPLAYED_PROXY_WARNING and self.proxy == 'legacy_requests' and ('http' in settings['proxies'] or 'https' in settings['proxies']):
        message = "'pywinrm' will use an environment defined proxy. This feature will be disabled in the future, please specify it explicitly."
        if 'http' in settings['proxies']:
            message += ' HTTP proxy {proxy} discovered.'.format(proxy=settings['proxies']['http'])
        if 'https' in settings['proxies']:
            message += ' HTTPS proxy {proxy} discovered.'.format(proxy=settings['proxies']['https'])
        DISPLAYED_PROXY_WARNING = True
        warnings.warn(message, DeprecationWarning)
    session.proxies = settings['proxies']
    session.verify = self.server_cert_validation == 'validate'
    if session.verify:
        if self.ca_trust_path == 'legacy_requests' and settings['verify'] is not None:
            session.verify = settings['verify']
            global DISPLAYED_CA_TRUST_WARNING
            if not DISPLAYED_CA_TRUST_WARNING and session.verify is not True:
                message = "'pywinrm' will use an environment variable defined CA Trust. This feature will be disabled in the future, please specify it explicitly."
                if os.environ.get('REQUESTS_CA_BUNDLE') is not None:
                    message += ' REQUESTS_CA_BUNDLE contains {ca_path}'.format(ca_path=os.environ.get('REQUESTS_CA_BUNDLE'))
                elif os.environ.get('CURL_CA_BUNDLE') is not None:
                    message += ' CURL_CA_BUNDLE contains {ca_path}'.format(ca_path=os.environ.get('CURL_CA_BUNDLE'))
                DISPLAYED_CA_TRUST_WARNING = True
                warnings.warn(message, DeprecationWarning)
        elif session.verify and self.ca_trust_path is not None:
            session.verify = self.ca_trust_path
    encryption_available = False
    if self.auth_method == 'kerberos':
        if not HAVE_KERBEROS:
            raise WinRMError('requested auth method is kerberos, but pykerberos is not installed')
        session.auth = HTTPKerberosAuth(mutual_authentication=REQUIRED, delegate=self.kerberos_delegation, force_preemptive=True, principal=self.username, hostname_override=self.kerberos_hostname_override, sanitize_mutual_error_response=False, service=self.service, send_cbt=self.send_cbt)
        encryption_available = hasattr(session.auth, 'winrm_encryption_available') and session.auth.winrm_encryption_available
    elif self.auth_method in ['certificate', 'ssl']:
        if self.auth_method == 'ssl' and (not self.cert_pem) and (not self.cert_key_pem):
            session.auth = requests.auth.HTTPBasicAuth(username=self.username, password=self.password)
        else:
            session.cert = (self.cert_pem, self.cert_key_pem)
            session.headers['Authorization'] = 'http://schemas.dmtf.org/wbem/wsman/1/wsman/secprofile/https/mutual'
    elif self.auth_method == 'ntlm':
        if not HAVE_NTLM:
            raise WinRMError('requested auth method is ntlm, but requests_ntlm is not installed')
        session.auth = HttpNtlmAuth(username=self.username, password=self.password, send_cbt=self.send_cbt)
        encryption_available = hasattr(session.auth, 'session_security')
    elif self.auth_method in ['basic', 'plaintext']:
        session.auth = requests.auth.HTTPBasicAuth(username=self.username, password=self.password)
    elif self.auth_method == 'credssp':
        if not HAVE_CREDSSP:
            raise WinRMError('requests auth method is credssp, but requests-credssp is not installed')
        session.auth = HttpCredSSPAuth(username=self.username, password=self.password, disable_tlsv1_2=self.credssp_disable_tlsv1_2, auth_mechanism=self.credssp_auth_mechanism, minimum_version=self.credssp_minimum_version)
        encryption_available = True
    else:
        raise WinRMError('unsupported auth method: %s' % self.auth_method)
    session.headers.update(self.default_headers)
    self.session = session
    if self.message_encryption == 'always' and (not encryption_available):
        raise WinRMError("message encryption is set to 'always' but the selected auth method %s does not support it" % self.auth_method)
    elif encryption_available:
        if self.message_encryption == 'always':
            self.setup_encryption()
        elif self.message_encryption == 'auto' and (not self.endpoint.lower().startswith('https')):
            self.setup_encryption()