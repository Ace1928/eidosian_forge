from __future__ import absolute_import
import logging
import os
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth.transport import _mtls_helper
from google.oauth2 import service_account
@property
def ssl_credentials(self):
    """Get the created SSL channel credentials.

        For devices with endpoint verification support, if the device certificate
        loading has any problems, corresponding exceptions will be raised. For
        a device without endpoint verification support, no exceptions will be
        raised.

        Returns:
            grpc.ChannelCredentials: The created grpc channel credentials.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS channel
                creation failed for any reason.
        """
    if self._is_mtls:
        try:
            _, cert, key, _ = _mtls_helper.get_client_ssl_credentials()
            self._ssl_credentials = grpc.ssl_channel_credentials(certificate_chain=cert, private_key=key)
        except exceptions.ClientCertError as caught_exc:
            new_exc = exceptions.MutualTLSChannelError(caught_exc)
            six.raise_from(new_exc, caught_exc)
    else:
        self._ssl_credentials = grpc.ssl_channel_credentials()
    return self._ssl_credentials