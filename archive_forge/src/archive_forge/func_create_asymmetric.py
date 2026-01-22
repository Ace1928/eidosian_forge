import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def create_asymmetric(self, name=None, algorithm=None, bit_length=None, pass_phrase=None, payload_content_type=None, expiration=None):
    """Factory method for `AsymmetricOrder` objects

        `AsymmetricOrder` objects returned by this method have not yet been
        submitted to the Barbican service.

        :param name: A friendly name for the container to be created
        :param algorithm: The algorithm associated with this secret key
        :param bit_length: The bit length of this secret key
        :param pass_phrase: Optional passphrase
        :param payload_content_type: The format/type of the secret data
        :param expiration: The expiration time of the secret in ISO 8601 format
        :returns: AsymmetricOrder
        :rtype: :class:`barbicanclient.v1.orders.AsymmetricOrder`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    return AsymmetricOrder(api=self._api, name=name, algorithm=algorithm, bit_length=bit_length, passphrase=pass_phrase, payload_content_type=payload_content_type, expiration=expiration)