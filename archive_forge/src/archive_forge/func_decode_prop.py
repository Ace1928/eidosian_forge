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
def decode_prop(self, prop, value):
    if isinstance(prop, ListProperty):
        return self.decode_list(prop, value)
    elif isinstance(prop, MapProperty):
        return self.decode_map(prop, value)
    else:
        return self.decode(prop.data_type, value)