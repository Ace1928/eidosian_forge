import datetime
import fnmatch
import mimetypes
import os
import stat as osstat  # os.stat constants
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.etag import calculate_multipart_etag
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
Really, "calculate md5", but since AWS uses their own format, we'll just call
    it a "local etag". TODO optimization: only calculate if remote key exists.