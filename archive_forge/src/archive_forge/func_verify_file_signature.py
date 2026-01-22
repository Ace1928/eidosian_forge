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
def verify_file_signature(manifest_file, detached_signature, keyring, ignore_signature_errors):
    """Run the gpg command and parse any errors. Raises CollectionSignatureError on failure."""
    gpg_result, gpg_verification_rc = run_gpg_verify(manifest_file, detached_signature, keyring, display)
    if gpg_result:
        errors = parse_gpg_errors(gpg_result)
        try:
            error = next(errors)
        except StopIteration:
            pass
        else:
            reasons = []
            ignored_reasons = 0
            for error in chain([error], errors):
                status_code = list(GPG_ERROR_MAP.keys())[list(GPG_ERROR_MAP.values()).index(error.__class__)]
                if status_code in ignore_signature_errors:
                    ignored_reasons += 1
                reasons.append(error.get_gpg_error_description())
            ignore = len(reasons) == ignored_reasons
            raise CollectionSignatureError(reasons=set(reasons), stdout=gpg_result, rc=gpg_verification_rc, ignore=ignore)
    if gpg_verification_rc:
        raise CollectionSignatureError(stdout=gpg_result, rc=gpg_verification_rc)
    return None