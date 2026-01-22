import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@staticmethod
def get_model_from_task(task: str, model_name_or_path: Union[str, Path], subfolder: str='', revision: Optional[str]=None, framework: Optional[str]=None, cache_dir: Optional[str]=None, torch_dtype: Optional['torch.dtype']=None, device: Optional[Union['torch.device', str]]=None, library_name: str=None, **model_kwargs) -> Union['PreTrainedModel', 'TFPreTrainedModel']:
    """
        Retrieves a model from its name and the task to be enabled.

        Args:
            task (`str`):
                The task required.
            model_name_or_path (`Union[str, Path]`):
                Can be either the model id of a model repo on the Hugging Face Hub, or a path to a local directory
                containing a model.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            framework (`Optional[str]`, *optional*):
                The framework to use for the export. See `TasksManager.determine_framework` for the priority should
                none be provided.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            torch_dtype (`Optional[torch.dtype]`, defaults to `None`):
                Data type to load the model on. PyTorch-only argument.
            device (`Optional[torch.device]`, defaults to `None`):
                Device to initialize the model on. PyTorch-only argument. For PyTorch, defaults to "cpu".
            model_kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to pass to the model `.from_pretrained()` method.
            library_name (`Optional[str]`, defaults to `None`):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers". See `TasksManager.infer_library_from_model` for the priority should
                none be provided.

        Returns:
            The instance of the model.

        """
    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)
    original_task = task
    if task == 'auto':
        task = TasksManager.infer_task_from_model(model_name_or_path, subfolder=subfolder, revision=revision)
    library_name = TasksManager.infer_library_from_model(model_name_or_path, subfolder, revision, cache_dir, library_name)
    model_type = None
    model_class_name = None
    kwargs = {'subfolder': subfolder, 'revision': revision, 'cache_dir': cache_dir, **model_kwargs}
    if library_name == 'transformers':
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        model_type = config.model_type.replace('_', '-')
        if original_task == 'automatic-speech-recognition' or task == 'automatic-speech-recognition':
            if original_task == 'auto' and config.architectures is not None:
                model_class_name = config.architectures[0]
    model_class = TasksManager.get_model_class_for_task(task, framework, model_type=model_type, model_class_name=model_class_name, library=library_name)
    if library_name == 'timm':
        model = model_class(f'hf_hub:{model_name_or_path}', pretrained=True, exportable=True)
        model = model.to(torch_dtype).to(device)
    elif library_name == 'sentence_transformers':
        cache_folder = model_kwargs.pop('cache_folder', None)
        use_auth_token = model_kwargs.pop('use_auth_token', None)
        trust_remote_code = model_kwargs.pop('trust_remote_code', False)
        model = model_class(model_name_or_path, device=device, cache_folder=cache_folder, use_auth_token=use_auth_token, trust_remote_code=trust_remote_code)
    else:
        try:
            if framework == 'pt':
                kwargs['torch_dtype'] = torch_dtype
                if isinstance(device, str):
                    device = torch.device(device)
                elif device is None:
                    device = torch.device('cpu')
                if version.parse(torch.__version__) >= version.parse('2.0') and library_name != 'diffusers':
                    with device:
                        model = model_class.from_pretrained(model_name_or_path, **kwargs)
                else:
                    model = model_class.from_pretrained(model_name_or_path, **kwargs).to(device)
            else:
                model = model_class.from_pretrained(model_name_or_path, **kwargs)
        except OSError:
            if framework == 'pt':
                logger.info('Loading TensorFlow model in PyTorch before exporting.')
                kwargs['from_tf'] = True
                model = model_class.from_pretrained(model_name_or_path, **kwargs)
            else:
                logger.info('Loading PyTorch model in TensorFlow before exporting.')
                kwargs['from_pt'] = True
                model = model_class.from_pretrained(model_name_or_path, **kwargs)
    TasksManager.standardize_model_attributes(model, library_name)
    return model