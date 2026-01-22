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
def publish_collection(collection_path, api, wait, timeout):
    """Publish an Ansible collection tarball into an Ansible Galaxy server.

    :param collection_path: The path to the collection tarball to publish.
    :param api: A GalaxyAPI to publish the collection to.
    :param wait: Whether to wait until the import process is complete.
    :param timeout: The time in seconds to wait for the import process to finish, 0 is indefinite.
    """
    import_uri = api.publish_collection(collection_path)
    if wait:
        task_id = None
        for path_segment in reversed(import_uri.split('/')):
            if path_segment:
                task_id = path_segment
                break
        if not task_id:
            raise AnsibleError("Publishing the collection did not return valid task info. Cannot wait for task status. Returned task info: '%s'" % import_uri)
        with _display_progress('Collection has been published to the Galaxy server {api.name!s} {api.api_server!s}'.format(api=api)):
            api.wait_import_task(task_id, timeout)
        display.display('Collection has been successfully published and imported to the Galaxy server %s %s' % (api.name, api.api_server))
    else:
        display.display('Collection has been pushed to the Galaxy server %s %s, not waiting until import has completed due to --no-wait being set. Import task results can be found at %s' % (api.name, api.api_server, import_uri))