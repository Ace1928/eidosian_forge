from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
@classmethod
def from_dir_path_as_unknown(cls, dir_path, art_mgr):
    """Make collection from an unspecified dir type.

        This alternative constructor attempts to grab metadata from the
        given path if it's a directory. If there's no metadata, it
        falls back to guessing the FQCN based on the directory path and
        sets the version to "*".

        It raises a ValueError immediately if the input is not an
        existing directory path.
        """
    if not os.path.isdir(dir_path):
        raise ValueError("The collection directory '{path!s}' doesn't exist".format(path=to_native(dir_path)))
    try:
        return cls.from_dir_path(dir_path, art_mgr)
    except ValueError:
        return cls.from_dir_path_implicit(dir_path)