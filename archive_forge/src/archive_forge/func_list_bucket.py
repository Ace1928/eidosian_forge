import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
def list_bucket(self, prefix='', delimiter='', headers=NOT_IMPL, all_versions=NOT_IMPL):
    return self.get_bucket().list(prefix=prefix, delimiter=delimiter)