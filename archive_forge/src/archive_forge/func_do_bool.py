import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def do_bool(val):
    return 'true' if val in [True, 1, '1', 'true'] else 'false'