import uuid
import base64
import time
from boto.compat import six, json
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.object import Object, StreamingObject
from boto.cloudfront.signers import ActiveTrustedSigners, TrustedSigners
from boto.cloudfront.logging import LoggingInfo
from boto.cloudfront.origin import S3Origin, CustomOrigin
from boto.s3.acl import ACL
def set_permissions_all(self, replace=False):
    """
        Sets the S3 ACL grants for all objects in the Distribution
        to the appropriate value based on the type of Distribution.

        :type replace: bool
        :param replace: If False, the Origin Access Identity will be
                        appended to the existing ACL for the object.
                        If True, the ACL for the object will be
                        completely replaced with one that grants
                        READ permission to the Origin Access Identity.

        """
    bucket = self._get_bucket()
    for key in bucket:
        self.set_permissions(key, replace)