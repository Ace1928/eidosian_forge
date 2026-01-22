import os
import zipfile
import logging
import tempfile
import uuid
import shutil
from ..utils import download, check_sha1, replace_file
from ... import base
Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    