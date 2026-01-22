import json
import os
from pathlib import Path
from pickle import DEFAULT_PROTOCOL, PicklingError
from typing import Any, Dict, List, Optional, Union
from packaging import version
from huggingface_hub import snapshot_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import (
from .utils import logging, validate_hf_hub_args
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility...
@validate_hf_hub_args
def push_to_hub_fastai(learner, *, repo_id: str, commit_message: str='Push FastAI model using huggingface_hub.', private: bool=False, token: Optional[str]=None, config: Optional[dict]=None, branch: Optional[str]=None, create_pr: Optional[bool]=None, allow_patterns: Optional[Union[List[str], str]]=None, ignore_patterns: Optional[Union[List[str], str]]=None, delete_patterns: Optional[Union[List[str], str]]=None, api_endpoint: Optional[str]=None):
    """
    Upload learner checkpoint files to the Hub.

    Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
    `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
    details.

    Args:
        learner (`Learner`):
            The `fastai.Learner' you'd like to push to the Hub.
        repo_id (`str`):
            The repository id for your model in Hub in the format of "namespace/repo_name". The namespace can be your individual account or an organization to which you have write access (for example, 'stanfordnlp/stanza-de').
        commit_message (`str`, *optional*):
            Message to commit while pushing. Will default to :obj:`"add model"`.
        private (`bool`, *optional*, defaults to `False`):
            Whether or not the repository created should be private.
        token (`str`, *optional*):
            The Hugging Face account token to use as HTTP bearer authorization for remote files. If :obj:`None`, the token will be asked by a prompt.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        branch (`str`, *optional*):
            The git branch on which to push the model. This defaults to
            the default branch as specified in your repository, which
            defaults to `"main"`.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `branch` with that commit.
            Defaults to `False`.
        api_endpoint (`str`, *optional*):
            The API endpoint to use when pushing the model to the hub.
        allow_patterns (`List[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are pushed.
        ignore_patterns (`List[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not pushed.
        delete_patterns (`List[str]` or `str`, *optional*):
            If provided, remote files matching any of the patterns will be deleted from the repo.

    Returns:
        The url of the commit of your model in the given repository.

    <Tip>

    Raises the following error:

        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          if the user is not log on to the Hugging Face Hub.

    </Tip>
    """
    _check_fastai_fastcore_versions()
    api = HfApi(endpoint=api_endpoint)
    repo_id = api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True).repo_id
    with SoftTemporaryDirectory() as tmp:
        saved_path = Path(tmp) / repo_id
        _save_pretrained_fastai(learner, saved_path, config=config)
        return api.upload_folder(repo_id=repo_id, token=token, folder_path=saved_path, commit_message=commit_message, revision=branch, create_pr=create_pr, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, delete_patterns=delete_patterns)