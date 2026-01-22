import copy
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .dynamic_module_utils import custom_object_save
from .tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (
@classmethod
def get_processor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    token = kwargs.pop('token', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', '')
    from_pipeline = kwargs.pop('_from_pipeline', None)
    from_auto_class = kwargs.pop('_from_auto', False)
    user_agent = {'file_type': 'processor', 'from_auto_class': from_auto_class}
    if from_pipeline is not None:
        user_agent['using_pipeline'] = from_pipeline
    if is_offline_mode() and (not local_files_only):
        logger.info('Offline mode: forcing local_files_only=True')
        local_files_only = True
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)
    if os.path.isfile(pretrained_model_name_or_path):
        resolved_processor_file = pretrained_model_name_or_path
        is_local = True
    elif is_remote_url(pretrained_model_name_or_path):
        processor_file = pretrained_model_name_or_path
        resolved_processor_file = download_url(pretrained_model_name_or_path)
    else:
        processor_file = PROCESSOR_NAME
        try:
            resolved_processor_file = cached_file(pretrained_model_name_or_path, processor_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, user_agent=user_agent, revision=revision, subfolder=subfolder, _raise_exceptions_for_missing_entries=False)
        except EnvironmentError:
            raise
        except Exception:
            raise EnvironmentError(f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory containing a {PROCESSOR_NAME} file")
    if resolved_processor_file is None:
        return ({}, kwargs)
    try:
        with open(resolved_processor_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        processor_dict = json.loads(text)
    except json.JSONDecodeError:
        raise EnvironmentError(f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file.")
    if is_local:
        logger.info(f'loading configuration file {resolved_processor_file}')
    else:
        logger.info(f'loading configuration file {processor_file} from cache at {resolved_processor_file}')
    if 'auto_map' in processor_dict and (not is_local):
        processor_dict['auto_map'] = add_model_info_to_auto_map(processor_dict['auto_map'], pretrained_model_name_or_path)
    return (processor_dict, kwargs)