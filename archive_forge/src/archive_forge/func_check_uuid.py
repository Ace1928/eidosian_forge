import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
def check_uuid(self, value):
    t = value.split('-')
    if len(t) != 5:
        raise ValueError