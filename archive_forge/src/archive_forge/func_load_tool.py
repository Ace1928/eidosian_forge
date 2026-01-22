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
def load_tool(task_or_repo_id, model_repo_id=None, remote=False, token=None, **kwargs):
    """
    Main function to quickly load a tool, be it on the Hub or in the Transformers library.

    Args:
        task_or_repo_id (`str`):
            The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers
            are:

            - `"document-question-answering"`
            - `"image-captioning"`
            - `"image-question-answering"`
            - `"image-segmentation"`
            - `"speech-to-text"`
            - `"summarization"`
            - `"text-classification"`
            - `"text-question-answering"`
            - `"text-to-speech"`
            - `"translation"`

        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        remote (`bool`, *optional*, defaults to `False`):
            Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        main_module = importlib.import_module('transformers')
        tools_module = main_module.tools
        tool_class = getattr(tools_module, tool_class_name)
        if remote:
            if model_repo_id is None:
                endpoints = get_default_endpoints()
                if task_or_repo_id not in endpoints:
                    raise ValueError(f'Could not infer a default endpoint for {task_or_repo_id}, you need to pass one using the `model_repo_id` argument.')
                model_repo_id = endpoints[task_or_repo_id]
            return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
        else:
            return tool_class(model_repo_id, token=token, **kwargs)
    else:
        return Tool.from_hub(task_or_repo_id, model_repo_id=model_repo_id, token=token, remote=remote, **kwargs)