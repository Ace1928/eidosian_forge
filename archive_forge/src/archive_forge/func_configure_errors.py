import os
from boto.compat import six
from datetime import datetime
import boto
from boto import config
from boto.compat import expanduser
from boto.pyami.config import Config
from boto.exception import InvalidInstanceMetadataError
from boto.gs.acl import ACL
from boto.gs.acl import CannedACLStrings as CannedGSACLStrings
from boto.s3.acl import CannedACLStrings as CannedS3ACLStrings
from boto.s3.acl import Policy
def configure_errors(self):
    error_map = self.ErrorMap[self.name]
    self.storage_copy_error = error_map[STORAGE_COPY_ERROR]
    self.storage_create_error = error_map[STORAGE_CREATE_ERROR]
    self.storage_data_error = error_map[STORAGE_DATA_ERROR]
    self.storage_permissions_error = error_map[STORAGE_PERMISSIONS_ERROR]
    self.storage_response_error = error_map[STORAGE_RESPONSE_ERROR]