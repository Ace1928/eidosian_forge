import string
from urllib.parse import urlparse
from ansible.module_utils.basic import to_text
def s3_head_objects(client, parts, bucket, obj, versionId):
    args = {'Bucket': bucket, 'Key': obj}
    if versionId:
        args['VersionId'] = versionId
    for part in range(1, parts + 1):
        args['PartNumber'] = part
        yield client.head_object(**args)