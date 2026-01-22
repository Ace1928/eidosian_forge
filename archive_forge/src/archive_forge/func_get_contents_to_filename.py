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
def get_contents_to_filename(self, filename, headers=None, cb=None, num_cb=10, torrent=False, version_id=None, res_download_handler=None, response_headers=None):
    """
        Retrieve an object from S3 using the name of the Key object as the
        key in S3.  Store contents of the object to a file named by 'filename'.
        See get_contents_to_file method for details about the
        parameters.

        :type filename: string
        :param filename: The filename of where to put the file contents

        :type headers: dict
        :param headers: Any additional headers to send in the request

        :type cb: function
        :param cb: a callback function that will be called to report
            progress on the upload.  The callback should accept two
            integer parameters, the first representing the number of
            bytes that have been successfully transmitted to S3 and
            the second representing the size of the to be transmitted
            object.

        :type num_cb: int
        :param num_cb: (optional) If a callback is specified with the
            cb parameter this parameter determines the granularity of
            the callback by defining the maximum number of times the
            callback will be called during the file transfer.

        :type torrent: bool
        :param torrent: If True, returns the contents of a torrent file
            as a string.

        :type res_upload_handler: ResumableDownloadHandler
        :param res_download_handler: If provided, this handler will
            perform the download.

        :type response_headers: dict
        :param response_headers: A dictionary containing HTTP
            headers/values that will override any headers associated
            with the stored object in the response.  See
            http://goo.gl/EWOPb for details.

        :type version_id: str
        :param version_id: The ID of a particular version of the object.
            If this parameter is not supplied but the Key object has
            a ``version_id`` attribute, that value will be used when
            retrieving the object.  You can set the Key object's
            ``version_id`` attribute to None to always grab the latest
            version from a version-enabled bucket.
        """
    try:
        with open(filename, 'wb') as fp:
            self.get_contents_to_file(fp, headers, cb, num_cb, torrent=torrent, version_id=version_id, res_download_handler=res_download_handler, response_headers=response_headers)
    except Exception:
        os.remove(filename)
        raise
    if self.last_modified is not None:
        try:
            modified_tuple = email.utils.parsedate_tz(self.last_modified)
            modified_stamp = int(email.utils.mktime_tz(modified_tuple))
            os.utime(fp.name, (modified_stamp, modified_stamp))
        except Exception:
            pass