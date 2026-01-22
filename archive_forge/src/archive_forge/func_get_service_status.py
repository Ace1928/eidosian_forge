from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def get_service_status(self, **kw):
    """Instruct the user on how to get service status.
        """
    sections = ', '.join(map(str.lower, api_version_path.keys()))
    message = 'Use {0}.get_(section)_service_status(), where (section) is one of the following: {1}'.format(self.__class__.__name__, sections)
    raise AttributeError(message)