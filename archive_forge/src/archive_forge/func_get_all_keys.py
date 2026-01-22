from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
def get_all_keys(self, headers=None, **params):
    """
        A lower-level method for listing contents of a bucket.  This
        closely models the actual S3 API and requires you to manually
        handle the paging of results.  For a higher-level method that
        handles the details of paging for you, you can use the list
        method.

        :type max_keys: int
        :param max_keys: The maximum number of keys to retrieve

        :type prefix: string
        :param prefix: The prefix of the keys you want to retrieve

        :type marker: string
        :param marker: The "marker" of where you are in the result set

        :type delimiter: string
        :param delimiter: If this optional, Unicode string parameter
            is included with your request, then keys that contain the
            same string between the prefix and the first occurrence of
            the delimiter will be rolled up into a single result
            element in the CommonPrefixes collection. These rolled-up
            keys are not returned elsewhere in the response.

        :param encoding_type: Requests Amazon S3 to encode the response and
            specifies the encoding method to use.

            An object key can contain any Unicode character; however, XML 1.0
            parser cannot parse some characters, such as characters with an
            ASCII value from 0 to 10. For characters that are not supported in
            XML 1.0, you can add this parameter to request that Amazon S3
            encode the keys in the response.

            Valid options: ``url``
        :type encoding_type: string

        :rtype: ResultSet
        :return: The result from S3 listing the keys requested

        """
    self.validate_kwarg_names(params, ['maxkeys', 'max_keys', 'prefix', 'marker', 'delimiter', 'encoding_type'])
    return self._get_all([('Contents', self.key_class), ('CommonPrefixes', Prefix)], '', headers, **params)