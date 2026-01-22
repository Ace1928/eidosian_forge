import xml.sax
import base64
import time
from boto.compat import six, urllib
from boto.auth import detect_potential_s3sigv4
import boto.utils
from boto.connection import AWSAuthConnection
from boto import handler
from boto.s3.bucket import Bucket
from boto.s3.key import Key
from boto.resultset import ResultSet
from boto.exception import BotoClientError, S3ResponseError
from boto.utils import get_utf8able_str
def head_bucket(self, bucket_name, headers=None):
    """
        Determines if a bucket exists by name.

        If the bucket does not exist, an ``S3ResponseError`` will be raised.

        :type bucket_name: string
        :param bucket_name: The name of the bucket

        :type headers: dict
        :param headers: Additional headers to pass along with the request to
            AWS.

        :returns: A <Bucket> object
        """
    response = self.make_request('HEAD', bucket_name, headers=headers)
    body = response.read()
    if response.status == 200:
        return self.bucket_class(self, bucket_name)
    elif response.status == 403:
        err = self.provider.storage_response_error(response.status, response.reason, body)
        err.error_code = 'AccessDenied'
        err.error_message = 'Access Denied'
        raise err
    elif response.status == 404:
        err = self.provider.storage_response_error(response.status, response.reason, body)
        err.error_code = 'NoSuchBucket'
        err.error_message = 'The specified bucket does not exist'
        raise err
    else:
        raise self.provider.storage_response_error(response.status, response.reason, body)