import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def get_key_rotation_status(self, key_id):
    """
        Retrieves a Boolean value that indicates whether key rotation
        is enabled for the specified key.

        :type key_id: string
        :param key_id: Unique identifier of the key. This can be an ARN, an
            alias, or a globally unique identifier.

        """
    params = {'KeyId': key_id}
    return self.make_request(action='GetKeyRotationStatus', body=json.dumps(params))