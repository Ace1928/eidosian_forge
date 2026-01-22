from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
@validate_hf_hub_args
def list_repo_tree(self, repo_id: str, path_in_repo: Optional[str]=None, *, recursive: bool=False, expand: bool=False, revision: Optional[str]=None, repo_type: Optional[str]=None, token: Optional[Union[bool, str]]=None) -> Iterable[Union[RepoFile, RepoFolder]]:
    """
        List a repo tree's files and folders and get information about them.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            path_in_repo (`str`, *optional*):
                Relative path of the tree (folder) in the repo, for example:
                `"checkpoints/1fec34a/results"`. Will default to the root tree (folder) of the repository.
            recursive (`bool`, *optional*, defaults to `False`):
                Whether to list tree's files and folders recursively.
            expand (`bool`, *optional*, defaults to `False`):
                Whether to fetch more information about the tree's files and folders (e.g. last commit and files' security scan results). This
                operation is more expensive for the server so only 50 results are returned per page (instead of 1000).
                As pagination is implemented in `huggingface_hub`, this is transparent for you except for the time it
                takes to get the results.
            revision (`str`, *optional*):
                The revision of the repository from which to get the tree. Defaults to `"main"` branch.
            repo_type (`str`, *optional*):
                The type of the repository from which to get the tree (`"model"`, `"dataset"` or `"space"`.
                Defaults to `"model"`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If `None` or `True` and
                machine is logged in (through `huggingface-cli login` or [`~huggingface_hub.login`]), token will be
                retrieved from the cache. If `False`, token is not sent in the request header.

        Returns:
            `Iterable[Union[RepoFile, RepoFolder]]`:
                The information about the tree's files and folders, as an iterable of [`RepoFile`] and [`RepoFolder`] objects. The order of the files and folders is
                not guaranteed.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.
            [`~utils.EntryNotFoundError`]:
                If the tree (folder) does not exist (error 404) on the repo.

        Examples:

            Get information about a repo's tree.
            ```py
            >>> from huggingface_hub import list_repo_tree
            >>> repo_tree = list_repo_tree("lysandre/arxiv-nlp")
            >>> repo_tree
            <generator object HfApi.list_repo_tree at 0x7fa4088e1ac0>
            >>> list(repo_tree)
            [
                RepoFile(path='.gitattributes', size=391, blob_id='ae8c63daedbd4206d7d40126955d4e6ab1c80f8f', lfs=None, last_commit=None, security=None),
                RepoFile(path='README.md', size=391, blob_id='43bd404b159de6fba7c2f4d3264347668d43af25', lfs=None, last_commit=None, security=None),
                RepoFile(path='config.json', size=554, blob_id='2f9618c3a19b9a61add74f70bfb121335aeef666', lfs=None, last_commit=None, security=None),
                RepoFile(
                    path='flax_model.msgpack', size=497764107, blob_id='8095a62ccb4d806da7666fcda07467e2d150218e',
                    lfs={'size': 497764107, 'sha256': 'd88b0d6a6ff9c3f8151f9d3228f57092aaea997f09af009eefd7373a77b5abb9', 'pointer_size': 134}, last_commit=None, security=None
                ),
                RepoFile(path='merges.txt', size=456318, blob_id='226b0752cac7789c48f0cb3ec53eda48b7be36cc', lfs=None, last_commit=None, security=None),
                RepoFile(
                    path='pytorch_model.bin', size=548123560, blob_id='64eaa9c526867e404b68f2c5d66fd78e27026523',
                    lfs={'size': 548123560, 'sha256': '9be78edb5b928eba33aa88f431551348f7466ba9f5ef3daf1d552398722a5436', 'pointer_size': 134}, last_commit=None, security=None
                ),
                RepoFile(path='vocab.json', size=898669, blob_id='b00361fece0387ca34b4b8b8539ed830d644dbeb', lfs=None, last_commit=None, security=None)]
            ]
            ```

            Get even more information about a repo's tree (last commit and files' security scan results)
            ```py
            >>> from huggingface_hub import list_repo_tree
            >>> repo_tree = list_repo_tree("prompthero/openjourney-v4", expand=True)
            >>> list(repo_tree)
            [
                RepoFolder(
                    path='feature_extractor',
                    tree_id='aa536c4ea18073388b5b0bc791057a7296a00398',
                    last_commit={
                        'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
                        'title': 'Upload diffusers weights (#48)',
                        'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
                    }
                ),
                RepoFolder(
                    path='safety_checker',
                    tree_id='65aef9d787e5557373fdf714d6c34d4fcdd70440',
                    last_commit={
                        'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
                        'title': 'Upload diffusers weights (#48)',
                        'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
                    }
                ),
                RepoFile(
                    path='model_index.json',
                    size=582,
                    blob_id='d3d7c1e8c3e78eeb1640b8e2041ee256e24c9ee1',
                    lfs=None,
                    last_commit={
                        'oid': 'b195ed2d503f3eb29637050a886d77bd81d35f0e',
                        'title': 'Fix deprecation warning by changing `CLIPFeatureExtractor` to `CLIPImageProcessor`. (#54)',
                        'date': datetime.datetime(2023, 5, 15, 21, 41, 59, tzinfo=datetime.timezone.utc)
                    },
                    security={
                        'safe': True,
                        'av_scan': {'virusFound': False, 'virusNames': None},
                        'pickle_import_scan': None
                    }
                )
                ...
            ]
            ```
        """
    repo_type = repo_type or REPO_TYPE_MODEL
    revision = quote(revision, safe='') if revision is not None else DEFAULT_REVISION
    headers = self._build_hf_headers(token=token)
    encoded_path_in_repo = '/' + quote(path_in_repo, safe='') if path_in_repo else ''
    tree_url = f'{self.endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}{encoded_path_in_repo}'
    for path_info in paginate(path=tree_url, headers=headers, params={'recursive': recursive, 'expand': expand}):
        yield (RepoFile(**path_info) if path_info['type'] == 'file' else RepoFolder(**path_info))