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
def connect_ec2containerservice(aws_access_key_id=None, aws_secret_access_key=None, **kwargs):
    """
    Connect to Amazon EC2 Container Service
    rtype: :class:`boto.ec2containerservice.layer1.EC2ContainerServiceConnection`
    :return: A connection to the Amazon EC2 Container Service
    """
    from boto.ec2containerservice.layer1 import EC2ContainerServiceConnection
    return EC2ContainerServiceConnection(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, **kwargs)