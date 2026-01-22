import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
from .agent_types import handle_agent_inputs, handle_agent_outputs
from {module_name} import {class_name}
def push_to_hub(self, repo_id: str, commit_message: str='Upload tool', private: Optional[bool]=None, token: Optional[Union[bool, str]]=None, create_pr: bool=False) -> str:
    """
        Upload the tool to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
    repo_url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type='space', space_sdk='gradio')
    repo_id = repo_url.repo_id
    metadata_update(repo_id, {'tags': ['tool']}, repo_type='space')
    with tempfile.TemporaryDirectory() as work_dir:
        self.save(work_dir)
        logger.info(f'Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}')
        return upload_folder(repo_id=repo_id, commit_message=commit_message, folder_path=work_dir, token=token, create_pr=create_pr, repo_type='space')