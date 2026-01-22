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
def encode_blob(self, value):
    if not value:
        return None
    if isinstance(value, six.string_types):
        return value
    if not value.id:
        bucket = self.manager.get_blob_bucket()
        key = bucket.new_key(str(uuid.uuid4()))
        value.id = 's3://%s/%s' % (key.bucket.name, key.name)
    else:
        match = re.match('^s3:\\/\\/([^\\/]*)\\/(.*)$', value.id)
        if match:
            s3 = self.manager.get_s3_connection()
            bucket = s3.get_bucket(match.group(1), validate=False)
            key = bucket.get_key(match.group(2))
        else:
            raise SDBPersistenceError('Invalid Blob ID: %s' % value.id)
    if value.value is not None:
        key.set_contents_from_string(value.value)
    return value.id