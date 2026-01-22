from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import xml.sax
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import GSResponseError
from boto.exception import InvalidAclError
from boto.gs.acl import ACL, CannedACLStrings
from boto.gs.acl import SupportedPermissions as GSPermissions
from boto.gs.bucketlistresultset import VersionedBucketListResultSet
from boto.gs.cors import Cors
from boto.gs.encryptionconfig import EncryptionConfig
from boto.gs.lifecycle import LifecycleConfig
from boto.gs.key import Key as GSKey
from boto.s3.acl import Policy
from boto.s3.bucket import Bucket as S3Bucket
from boto.utils import get_utf8able_str
from boto.compat import quote
from boto.compat import six
def list_versions(self, prefix='', delimiter='', marker='', generation_marker='', headers=None):
    """
        List versioned objects within a bucket.  This returns an
        instance of an VersionedBucketListResultSet that automatically
        handles all of the result paging, etc. from GCS.  You just need
        to keep iterating until there are no more results.  Called
        with no arguments, this will return an iterator object across
        all keys within the bucket.

        :type prefix: string
        :param prefix: allows you to limit the listing to a particular
            prefix.  For example, if you call the method with
            prefix='/foo/' then the iterator will only cycle through
            the keys that begin with the string '/foo/'.

        :type delimiter: string
        :param delimiter: can be used in conjunction with the prefix
            to allow you to organize and browse your keys
            hierarchically. See:
            https://developers.google.com/storage/docs/reference-headers#delimiter
            for more details.

        :type marker: string
        :param marker: The "marker" of where you are in the result set

        :type generation_marker: string
        :param generation_marker: The "generation marker" of where you are in
            the result set.

        :type headers: dict
        :param headers: A dictionary of header name/value pairs.

        :rtype:
            :class:`boto.gs.bucketlistresultset.VersionedBucketListResultSet`
        :return: an instance of a BucketListResultSet that handles paging, etc.
        """
    return VersionedBucketListResultSet(self, prefix, delimiter, marker, generation_marker, headers)