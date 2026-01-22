import datetime
import fnmatch
import mimetypes
import os
import stat as osstat  # os.stat constants
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.etag import calculate_multipart_etag
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def calculate_s3_path(filelist, key_prefix=''):
    ret = []
    for fileentry in filelist:
        retentry = fileentry.copy()
        retentry['s3_path'] = os.path.join(key_prefix, fileentry['chopped_path'])
        ret.append(retentry)
    return ret