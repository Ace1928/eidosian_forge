import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
def s3tolocal(self, s3path, bkt):
    import boto
    local_directory = str(self.inputs.local_directory)
    bucket_path = str(self.inputs.bucket_path)
    template = str(self.inputs.template)
    if not os.path.basename(local_directory) == '':
        local_directory += '/'
    if not os.path.basename(bucket_path) == '':
        bucket_path += '/'
    if template[0] == '/':
        template = template[1:]
    localpath = s3path.replace(bucket_path, local_directory)
    localdir = os.path.split(localpath)[0]
    if not os.path.exists(localdir):
        os.makedirs(localdir)
    k = boto.s3.key.Key(bkt)
    k.key = s3path
    k.get_contents_to_filename(localpath)
    return localpath