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
def verify_file_signatures(fqcn, manifest_file, detached_signatures, keyring, required_successful_count, ignore_signature_errors):
    successful = 0
    error_messages = []
    signature_count_requirements = re.match(SIGNATURE_COUNT_RE, required_successful_count).groupdict()
    strict = signature_count_requirements['strict'] or False
    require_all = signature_count_requirements['all']
    require_count = signature_count_requirements['count']
    if require_count is not None:
        require_count = int(require_count)
    for signature in detached_signatures:
        signature = to_text(signature, errors='surrogate_or_strict')
        try:
            verify_file_signature(manifest_file, signature, keyring, ignore_signature_errors)
        except CollectionSignatureError as error:
            if error.ignore:
                continue
            error_messages.append(error.report(fqcn))
        else:
            successful += 1
            if require_all:
                continue
            if successful == require_count:
                break
    if strict and (not successful):
        verified = False
        display.display(f"Signature verification failed for '{fqcn}': no successful signatures")
    elif require_all:
        verified = not error_messages
        if not verified:
            display.display(f"Signature verification failed for '{fqcn}': some signatures failed")
    else:
        verified = not detached_signatures or require_count == successful
        if not verified:
            display.display(f"Signature verification failed for '{fqcn}': fewer successful signatures than required")
    if not verified:
        for msg in error_messages:
            display.vvvv(msg)
    return verified