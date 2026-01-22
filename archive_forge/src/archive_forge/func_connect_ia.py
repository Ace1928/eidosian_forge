from boto.pyami.config import Config, BotoConfigLocations
from boto.storage_uri import BucketStorageUri, FileStorageUri
import boto.plugin
import datetime
import os
import platform
import re
import sys
import logging
import logging.config
from boto.compat import urlparse
from boto.exception import InvalidUriError
def connect_ia(ia_access_key_id=None, ia_secret_access_key=None, is_secure=False, **kwargs):
    """
    Connect to the Internet Archive via their S3-like API.

    :type ia_access_key_id: string
    :param ia_access_key_id: Your IA Access Key ID.  This will also look
        in your boto config file for an entry in the Credentials
        section called "ia_access_key_id"

    :type ia_secret_access_key: string
    :param ia_secret_access_key: Your IA Secret Access Key.  This will also
        look in your boto config file for an entry in the Credentials
        section called "ia_secret_access_key"

    :rtype: :class:`boto.s3.connection.S3Connection`
    :return: A connection to the Internet Archive
    """
    from boto.s3.connection import S3Connection
    from boto.s3.connection import OrdinaryCallingFormat
    access_key = config.get('Credentials', 'ia_access_key_id', ia_access_key_id)
    secret_key = config.get('Credentials', 'ia_secret_access_key', ia_secret_access_key)
    return S3Connection(access_key, secret_key, host='s3.us.archive.org', calling_format=OrdinaryCallingFormat(), is_secure=is_secure, **kwargs)