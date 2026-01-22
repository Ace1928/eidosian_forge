import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def encode_reference(self, value):
    if value in (None, 'None', '', ' '):
        return None
    if isinstance(value, six.string_types):
        return value
    else:
        return value.id