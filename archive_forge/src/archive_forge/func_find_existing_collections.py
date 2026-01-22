from __future__ import (absolute_import, division, print_function)
import errno
import fnmatch
import functools
import json
import os
import pathlib
import queue
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import typing as t
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from hashlib import sha256
from io import BytesIO
from importlib.metadata import distribution
from itertools import chain
import ansible.constants as C
from ansible.compat.importlib_resources import files
from ansible.errors import AnsibleError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.collection.gpg import (
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import meets_requirements
from ansible.plugins.loader import get_all_plugin_loaders
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_dump
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash, secure_hash_s
from ansible.utils.sentinel import Sentinel
def find_existing_collections(path_filter, artifacts_manager, namespace_filter=None, collection_filter=None, dedupe=True):
    """Locate all collections under a given path.

    :param path: Collection dirs layout search path.
    :param artifacts_manager: Artifacts manager.
    """
    if files is None:
        raise AnsibleError('importlib_resources is not installed and is required')
    if path_filter and (not is_sequence(path_filter)):
        path_filter = [path_filter]
    if namespace_filter and (not is_sequence(namespace_filter)):
        namespace_filter = [namespace_filter]
    if collection_filter and (not is_sequence(collection_filter)):
        collection_filter = [collection_filter]
    paths = set()
    for path in files('ansible_collections').glob('*/*/'):
        path = _normalize_collection_path(path)
        if not path.is_dir():
            continue
        if path_filter:
            for pf in path_filter:
                try:
                    path.relative_to(_normalize_collection_path(pf))
                except ValueError:
                    continue
                break
            else:
                continue
        paths.add(path)
    seen = set()
    for path in paths:
        namespace = path.parent.name
        name = path.name
        if namespace_filter and namespace not in namespace_filter:
            continue
        if collection_filter and name not in collection_filter:
            continue
        if dedupe:
            try:
                collection_path = files(f'ansible_collections.{namespace}.{name}')
            except ImportError:
                continue
            if collection_path in seen:
                continue
            seen.add(collection_path)
        else:
            collection_path = path
        b_collection_path = to_bytes(collection_path.as_posix())
        try:
            req = Candidate.from_dir_path_as_unknown(b_collection_path, artifacts_manager)
        except ValueError as val_err:
            display.warning(f'{val_err}')
            continue
        display.vvv(u"Found installed collection {coll!s} at '{path!s}'".format(coll=to_text(req), path=to_text(req.src)))
        yield req