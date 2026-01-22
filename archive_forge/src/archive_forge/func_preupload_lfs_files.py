import time
from functools import partial
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from packaging import version
from requests import ConnectionError, HTTPError
from .. import config
from . import logging
def preupload_lfs_files(hf_api: HfApi, **kwargs):
    hf_api.preupload_lfs_files(**kwargs)