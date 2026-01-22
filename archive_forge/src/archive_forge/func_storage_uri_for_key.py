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
def storage_uri_for_key(key):
    """Returns a StorageUri for the given key.

    :type key: :class:`boto.s3.key.Key` or subclass
    :param key: URI naming bucket + optional object.
    """
    if not isinstance(key, boto.s3.key.Key):
        raise InvalidUriError('Requested key (%s) is not a subclass of boto.s3.key.Key' % str(type(key)))
    prov_name = key.bucket.connection.provider.get_provider_name()
    uri_str = '%s://%s/%s' % (prov_name, key.bucket.name, key.name)
    return storage_uri(uri_str)