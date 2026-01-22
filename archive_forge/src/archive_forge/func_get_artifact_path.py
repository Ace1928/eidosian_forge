from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
def get_artifact_path(self, collection):
    """Given a concrete collection pointer, return a cached path.

        If it's not yet on disk, this method downloads the artifact first.
        """
    try:
        return self._artifact_cache[collection.src]
    except KeyError:
        pass
    if collection.is_url:
        display.vvvv("Collection requirement '{collection!s}' is a URL to a tar artifact".format(collection=collection.fqcn))
        try:
            b_artifact_path = _download_file(collection.src, self._b_working_directory, expected_hash=None, validate_certs=self._validate_certs, timeout=self.timeout)
        except Exception as err:
            raise AnsibleError("Failed to download collection tar from '{coll_src!s}': {download_err!s}".format(coll_src=to_native(collection.src), download_err=to_native(err))) from err
    elif collection.is_scm:
        b_artifact_path = _extract_collection_from_git(collection.src, collection.ver, self._b_working_directory)
    elif collection.is_file or collection.is_dir or collection.is_subdirs:
        b_artifact_path = to_bytes(collection.src)
    else:
        raise RuntimeError('The artifact is of an unexpected type {art_type!s}'.format(art_type=collection.type))
    self._artifact_cache[collection.src] = b_artifact_path
    return b_artifact_path