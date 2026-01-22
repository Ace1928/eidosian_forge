import inspect
import io
import os
import re
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import unquote
from huggingface_hub.constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, REPO_TYPES_URL_PREFIXES
from huggingface_hub.utils import get_session
from .utils import (
from .utils.sha import sha256, sha_fileobj
def lfs_upload(operation: 'CommitOperationAdd', lfs_batch_action: Dict, token: Optional[str]) -> None:
    """
    Handles uploading a given object to the Hub with the LFS protocol.

    Can be a No-op if the content of the file is already present on the hub large file storage.

    Args:
        operation (`CommitOperationAdd`):
            The add operation triggering this upload.
        lfs_batch_action (`dict`):
            Upload instructions from the LFS batch endpoint for this object. See [`~utils.lfs.post_lfs_batch_info`] for
            more details.
        token (`str`, *optional*):
            A [user access token](https://hf.co/settings/tokens) to authenticate requests against the Hub

    Raises:
        - `ValueError` if `lfs_batch_action` is improperly formatted
        - `HTTPError` if the upload resulted in an error
    """
    _validate_batch_actions(lfs_batch_action)
    actions = lfs_batch_action.get('actions')
    if actions is None:
        logger.debug(f'Content of file {operation.path_in_repo} is already present upstream - skipping upload')
        return
    upload_action = lfs_batch_action['actions']['upload']
    _validate_lfs_action(upload_action)
    verify_action = lfs_batch_action['actions'].get('verify')
    if verify_action is not None:
        _validate_lfs_action(verify_action)
    header = upload_action.get('header', {})
    chunk_size = header.get('chunk_size')
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
        except (ValueError, TypeError):
            raise ValueError(f"Malformed response from LFS batch endpoint: `chunk_size` should be an integer. Got '{chunk_size}'.")
        _upload_multi_part(operation=operation, header=header, chunk_size=chunk_size, upload_url=upload_action['href'])
    else:
        _upload_single_part(operation=operation, upload_url=upload_action['href'])
    if verify_action is not None:
        _validate_lfs_action(verify_action)
        verify_resp = get_session().post(verify_action['href'], headers=build_hf_headers(token=token or True), json={'oid': operation.upload_info.sha256.hex(), 'size': operation.upload_info.size})
        hf_raise_for_status(verify_resp)
    logger.debug(f'{operation.path_in_repo}: Upload successful')