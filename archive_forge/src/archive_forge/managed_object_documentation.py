import abc
import binascii
from castellan.common import exception
Returns a dict that can be used with the from_dict() method.

        :param metadata_only: A switch to create an dictionary with metadata
                              only, without the secret itself.

        :rtype: dict
        