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
def encode_map(self, prop, value):
    import urllib
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Expected a dict value, got %s' % type(value))
    new_value = []
    for key in value:
        item_type = getattr(prop, 'item_type')
        if self.model_class in item_type.mro():
            item_type = self.model_class
        encoded_value = self.encode(item_type, value[key])
        if encoded_value is not None:
            new_value.append('%s:%s' % (urllib.quote(key), encoded_value))
    return new_value