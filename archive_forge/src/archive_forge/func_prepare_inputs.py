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
def prepare_inputs(self, *args, **kwargs):
    """
        Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be
        matched with the signature of the `tool_class` if it was provided at instantation. Images will be encoded into
        bytes.

        You can override this method in your custom class of [`RemoteTool`].
        """
    inputs = kwargs.copy()
    if len(args) > 0:
        if self.tool_class is not None:
            if issubclass(self.tool_class, PipelineTool):
                call_method = self.tool_class.encode
            else:
                call_method = self.tool_class.__call__
            signature = inspect.signature(call_method).parameters
            parameters = [k for k, p in signature.items() if p.kind not in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]]
            if parameters[0] == 'self':
                parameters = parameters[1:]
            if len(args) > len(parameters):
                raise ValueError(f'{self.tool_class} only accepts {len(parameters)} arguments but {len(args)} were given.')
            for arg, name in zip(args, parameters):
                inputs[name] = arg
        elif len(args) > 1:
            raise ValueError('A `RemoteTool` can only accept one positional input.')
        elif len(args) == 1:
            if is_pil_image(args[0]):
                return {'inputs': self.client.encode_image(args[0])}
            return {'inputs': args[0]}
    for key, value in inputs.items():
        if is_pil_image(value):
            inputs[key] = self.client.encode_image(value)
    return {'inputs': inputs}