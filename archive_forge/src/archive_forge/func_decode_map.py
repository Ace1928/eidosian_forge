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
def decode_map(self, prop, value):
    if not isinstance(value, list):
        value = [value]
    ret_value = {}
    item_type = getattr(prop, 'item_type')
    for val in value:
        k, v = self.decode_map_element(item_type, val)
        ret_value[k] = v
    return ret_value