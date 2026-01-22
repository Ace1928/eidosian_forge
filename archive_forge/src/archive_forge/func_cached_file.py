import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def cached_file(path_or_repo_id: Union[str, os.PathLike], filename: str, cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=False, subfolder: str='', repo_type: Optional[str]=None, user_agent: Optional[Union[str, Dict[str, str]]]=None, _raise_exceptions_for_gated_repo: bool=True, _raise_exceptions_for_missing_entries: bool=True, _raise_exceptions_for_connection_errors: bool=True, _commit_hash: Optional[str]=None, **deprecated_kwargs) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
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
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
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
    if subfolder is None:
        subfolder = ''
    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout 'https://huggingface.co/{path_or_repo_id}/{revision}' for available files.")
            else:
                return None
        return resolved_file
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if _commit_hash is not None and (not force_download):
        resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type)
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(f'Could not locate {full_filename} inside {path_or_repo_id}.')
    user_agent = http_user_agent(user_agent)
    try:
        resolved_file = hf_hub_download(path_or_repo_id, filename, subfolder=None if len(subfolder) == 0 else subfolder, repo_type=repo_type, revision=revision, cache_dir=cache_dir, user_agent=user_agent, force_download=force_download, proxies=proxies, resume_download=resume_download, token=token, local_files_only=local_files_only)
    except GatedRepoError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_gated_repo:
            return resolved_file
        raise EnvironmentError(f'You are trying to access a gated repo.\nMake sure to have access to it at https://huggingface.co/{path_or_repo_id}.\n{str(e)}') from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(f"{path_or_repo_id} is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`") from e
    except RevisionNotFoundError as e:
        raise EnvironmentError(f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this model name. Check the model page at 'https://huggingface.co/{path_or_repo_id}' for available revisions.") from e
    except LocalEntryNotFoundError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_missing_entries or (not _raise_exceptions_for_connection_errors):
            return resolved_file
        raise EnvironmentError(f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.") from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = 'main'
        raise EnvironmentError(f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout 'https://huggingface.co/{path_or_repo_id}/{revision}' for available files.") from e
    except HTTPError as err:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_connection_errors:
            return resolved_file
        raise EnvironmentError(f'There was a specific connection error when trying to load {path_or_repo_id}:\n{err}')
    except HFValidationError as e:
        raise EnvironmentError(f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a local folder or the repo_id of a model on the Hub.") from e
    return resolved_file