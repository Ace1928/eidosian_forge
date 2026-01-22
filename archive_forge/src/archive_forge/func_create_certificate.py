import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def create_certificate(self, name=None, certificate=None, intermediates=None, private_key=None, private_key_passphrase=None):
    """Factory method for `CertificateContainer` objects

        `CertificateContainer` objects returned by this method have not yet
        been stored in Barbican.

        :param name: A friendly name for the CertificateContainer
        :param certificate: Secret object containing a Certificate
        :param intermediates: Secret object containing Intermediate Certs
        :param private_key: Secret object containing a Private Key
        :param private_key_passphrase: Secret object containing a passphrase
        :returns: CertificateContainer
        :rtype: :class:`barbicanclient.v1.containers.CertificateContainer`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    return CertificateContainer(api=self._api, name=name, certificate=certificate, intermediates=intermediates, private_key=private_key, private_key_passphrase=private_key_passphrase)