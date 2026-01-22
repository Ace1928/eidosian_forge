import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_tag_resource(self, resource, tag):
    """
        Associate tag with the provided resource.

        :param resource: Resource to associate a tag with.
        :type resource: :class:`libcloud.compute.base.Node` or
                        :class:`.CloudSigmaDrive`

        :param tag: Tag to associate with the resources.
        :type tag: :class:`.CloudSigmaTag`

        :return: Updated tag object.
        :rtype: :class:`.CloudSigmaTag`
        """
    if not hasattr(resource, 'id'):
        raise ValueError("Resource doesn't have id attribute")
    return self.ex_tag_resources(resources=[resource], tag=tag)