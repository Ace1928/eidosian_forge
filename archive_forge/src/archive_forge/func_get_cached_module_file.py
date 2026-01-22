import filecmp
import importlib
import os
import re
import shutil
import signal
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import try_to_load_from_cache
from .utils import (
def get_cached_module_file(pretrained_model_name_or_path: Union[str, os.PathLike], module_file: str, cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=False, repo_type: Optional[str]=None, _commit_hash: Optional[str]=None, **deprecated_kwargs) -> str:
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    use_auth_token = deprecated_kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    if is_offline_mode() and (not local_files_only):
        logger.info('Offline mode: forcing local_files_only=True')
        local_files_only = True
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        submodule = os.path.basename(pretrained_model_name_or_path)
    else:
        submodule = pretrained_model_name_or_path.replace('/', os.path.sep)
        cached_module = try_to_load_from_cache(pretrained_model_name_or_path, module_file, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type)
    new_files = []
    try:
        resolved_module_file = cached_file(pretrained_model_name_or_path, module_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, revision=revision, repo_type=repo_type, _commit_hash=_commit_hash)
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)
    except EnvironmentError:
        logger.error(f'Could not locate the {module_file} inside {pretrained_model_name_or_path}.')
        raise
    modules_needed = check_imports(resolved_module_file)
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    if submodule == os.path.basename(pretrained_model_name_or_path):
        if not (submodule_path / module_file).exists() or not filecmp.cmp(resolved_module_file, str(submodule_path / module_file)):
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        for module_needed in modules_needed:
            module_needed = f'{module_needed}.py'
            module_needed_file = os.path.join(pretrained_model_name_or_path, module_needed)
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(module_needed_file, str(submodule_path / module_needed)):
                shutil.copy(module_needed_file, submodule_path / module_needed)
                importlib.invalidate_caches()
    else:
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        create_dynamic_module(full_submodule)
        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        for module_needed in modules_needed:
            if not (submodule_path / f'{module_needed}.py').exists():
                get_cached_module_file(pretrained_model_name_or_path, f'{module_needed}.py', cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, token=token, revision=revision, local_files_only=local_files_only, _commit_hash=commit_hash)
                new_files.append(f'{module_needed}.py')
    if len(new_files) > 0 and revision is None:
        new_files = '\n'.join([f'- {f}' for f in new_files])
        repo_type_str = '' if repo_type is None else f'{repo_type}s/'
        url = f'https://huggingface.co/{repo_type_str}{pretrained_model_name_or_path}'
        logger.warning(f'A new version of the following files was downloaded from {url}:\n{new_files}\n. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.')
    return os.path.join(full_submodule, module_file)