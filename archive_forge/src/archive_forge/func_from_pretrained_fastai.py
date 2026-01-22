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
def from_pretrained_fastai(repo_id: str, revision: Optional[str]=None):
    """
    Load pretrained fastai model from the Hub or from a local directory.

    Args:
        repo_id (`str`):
            The location where the pickled fastai.Learner is. It can be either of the two:
                - Hosted on the Hugging Face Hub. E.g.: 'espejelomar/fatai-pet-breeds-classification' or 'distilgpt2'.
                  You can add a `revision` by appending `@` at the end of `repo_id`. E.g.: `dbmdz/bert-base-german-cased@main`.
                  Revision is the specific model version to use. Since we use a git-based system for storing models and other
                  artifacts on the Hugging Face Hub, it can be a branch name, a tag name, or a commit id.
                - Hosted locally. `repo_id` would be a directory containing the pickle and a pyproject.toml
                  indicating the fastai and fastcore versions used to build the `fastai.Learner`. E.g.: `./my_model_directory/`.
        revision (`str`, *optional*):
            Revision at which the repo's files are downloaded. See documentation of `snapshot_download`.

    Returns:
        The `fastai.Learner` model in the `repo_id` repo.
    """
    _check_fastai_fastcore_versions()
    if not os.path.isdir(repo_id):
        storage_folder = snapshot_download(repo_id=repo_id, revision=revision, library_name='fastai', library_version=get_fastai_version())
    else:
        storage_folder = repo_id
    _check_fastai_fastcore_pyproject_versions(storage_folder)
    from fastai.learner import load_learner
    return load_learner(os.path.join(storage_folder, 'model.pkl'))