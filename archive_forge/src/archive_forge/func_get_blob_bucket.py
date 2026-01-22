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
def get_blob_bucket(self, bucket_name=None):
    s3 = self.get_s3_connection()
    bucket_name = '%s-%s' % (s3.aws_access_key_id, self.domain.name)
    bucket_name = bucket_name.lower()
    try:
        self.bucket = s3.get_bucket(bucket_name)
    except:
        self.bucket = s3.create_bucket(bucket_name)
    return self.bucket