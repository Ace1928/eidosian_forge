from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def generate_url(self, expires_in, method='GET', headers=None, query_auth=True, force_http=False, response_headers=None, expires_in_absolute=False, version_id=None, policy=None, reduced_redundancy=False, encrypt_key=False):
    """
        Generate a URL to access this key.

        :type expires_in: int
        :param expires_in: How long the url is valid for, in seconds.

        :type method: string
        :param method: The method to use for retrieving the file
            (default is GET).

        :type headers: dict
        :param headers: Any headers to pass along in the request.

        :type query_auth: bool
        :param query_auth: If True, signs the request in the URL.

        :type force_http: bool
        :param force_http: If True, http will be used instead of https.

        :type response_headers: dict
        :param response_headers: A dictionary containing HTTP
            headers/values that will override any headers associated
            with the stored object in the response.  See
            http://goo.gl/EWOPb for details.

        :type expires_in_absolute: bool
        :param expires_in_absolute:

        :type version_id: string
        :param version_id: The version_id of the object to GET. If specified
            this overrides any value in the key.

        :type policy: :class:`boto.s3.acl.CannedACLStrings`
        :param policy: A canned ACL policy that will be applied to the
            new key in S3.

        :type reduced_redundancy: bool
        :param reduced_redundancy: If True, this will set the storage
            class of the new Key to be REDUCED_REDUNDANCY. The Reduced
            Redundancy Storage (RRS) feature of S3, provides lower
            redundancy at lower storage cost.

        :type encrypt_key: bool
        :param encrypt_key: If True, the new copy of the object will
            be encrypted on the server-side by S3 and will be stored
            in an encrypted form while at rest in S3.

        :rtype: string
        :return: The URL to access the key
        """
    provider = self.bucket.connection.provider
    version_id = version_id or self.version_id
    if headers is None:
        headers = {}
    else:
        headers = headers.copy()
    if policy:
        headers[provider.acl_header] = policy
    if reduced_redundancy:
        self.storage_class = 'REDUCED_REDUNDANCY'
        if provider.storage_class_header:
            headers[provider.storage_class_header] = self.storage_class
    if encrypt_key:
        headers[provider.server_side_encryption_header] = 'AES256'
    headers = boto.utils.merge_meta(headers, self.metadata, provider)
    return self.bucket.connection.generate_url(expires_in, method, self.bucket.name, self.name, headers, query_auth, force_http, response_headers, expires_in_absolute, version_id)