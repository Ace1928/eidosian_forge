import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def delete_alias(self, alias_name):
    """
        Deletes the specified alias.

        :type alias_name: string
        :param alias_name: The alias to be deleted.

        """
    params = {'AliasName': alias_name}
    return self.make_request(action='DeleteAlias', body=json.dumps(params))