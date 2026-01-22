import os
import zipfile
import logging
import tempfile
import uuid
import shutil
from ..utils import download, check_sha1, replace_file
from ... import base
def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]