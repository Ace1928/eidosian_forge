import datetime
import fnmatch
import mimetypes
import os
import stat as osstat  # os.stat constants
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.etag import calculate_multipart_etag
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def determine_mimetypes(filelist, override_map):
    ret = []
    for fileentry in filelist:
        retentry = fileentry.copy()
        localfile = fileentry['fullpath']
        file_extension = os.path.splitext(localfile)[1]
        if override_map and override_map.get(file_extension):
            retentry['mime_type'] = override_map[file_extension]
        else:
            retentry['mime_type'], retentry['encoding'] = mimetypes.guess_type(localfile, strict=False)
        if not retentry['mime_type']:
            retentry['mime_type'] = 'application/octet-stream'
        ret.append(retentry)
    return ret