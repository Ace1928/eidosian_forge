import os
import boto
from boto.utils import get_instance_metadata, get_instance_userdata
from boto.pyami.config import Config, BotoConfigPath
from boto.pyami.scriptbase import ScriptBase
import time
def fetch_s3_file(self, s3_file):
    try:
        from boto.utils import fetch_file
        f = fetch_file(s3_file)
        path = os.path.join(self.working_dir, s3_file.split('/')[-1])
        open(path, 'w').write(f.read())
    except:
        boto.log.exception('Problem Retrieving file: %s' % s3_file)
        path = None
    return path