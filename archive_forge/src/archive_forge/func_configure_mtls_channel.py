from __future__ import absolute_import
import functools
import logging
import numbers
import os
import time
import requests.adapters  # pylint: disable=ungrouped-imports
import requests.exceptions  # pylint: disable=ungrouped-imports
from requests.packages.urllib3.util.ssl_ import (  # type: ignore
import six  # pylint: disable=ungrouped-imports
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
import google.auth.transport._mtls_helper
from google.oauth2 import service_account
def configure_mtls_channel(self, client_cert_callback=None):
    """Configure the client certificate and key for SSL connection.

        The function does nothing unless `GOOGLE_API_USE_CLIENT_CERTIFICATE` is
        explicitly set to `true`. In this case if client certificate and key are
        successfully obtained (from the given client_cert_callback or from application
        default SSL credentials), a :class:`_MutualTlsAdapter` instance will be mounted
        to "https://" prefix.

        Args:
            client_cert_callback (Optional[Callable[[], (bytes, bytes)]]):
                The optional callback returns the client certificate and private
                key bytes both in PEM format.
                If the callback is None, application default SSL credentials
                will be used.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
    use_client_cert = os.getenv(environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE, 'false')
    if use_client_cert != 'true':
        self._is_mtls = False
        return
    try:
        import OpenSSL
    except ImportError as caught_exc:
        new_exc = exceptions.MutualTLSChannelError(caught_exc)
        six.raise_from(new_exc, caught_exc)
    try:
        self._is_mtls, cert, key = google.auth.transport._mtls_helper.get_client_cert_and_key(client_cert_callback)
        if self._is_mtls:
            mtls_adapter = _MutualTlsAdapter(cert, key)
            self.mount('https://', mtls_adapter)
    except (exceptions.ClientCertError, ImportError, OpenSSL.crypto.Error) as caught_exc:
        new_exc = exceptions.MutualTLSChannelError(caught_exc)
        six.raise_from(new_exc, caught_exc)