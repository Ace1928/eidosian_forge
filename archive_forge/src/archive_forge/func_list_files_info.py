import time
from functools import partial
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from packaging import version
from requests import ConnectionError, HTTPError
from .. import config
from . import logging
def list_files_info(hf_api: HfApi, **kwargs):
    kwargs = {**kwargs, 'recursive': True}
    for repo_path in hf_api.list_repo_tree(**kwargs):
        if isinstance(repo_path, RepoFile):
            yield repo_path