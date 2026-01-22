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
def from_dir_path(cls, dir_path, art_mgr):
    """Make collection from an directory with metadata."""
    if dir_path.endswith(to_bytes(os.path.sep)):
        dir_path = dir_path.rstrip(to_bytes(os.path.sep))
    if not _is_collection_dir(dir_path):
        display.warning(u"Collection at '{path!s}' does not have a {manifest_json!s} file, nor has it {galaxy_yml!s}: cannot detect version.".format(galaxy_yml=to_text(_GALAXY_YAML), manifest_json=to_text(_MANIFEST_JSON), path=to_text(dir_path, errors='surrogate_or_strict')))
        raise ValueError('`dir_path` argument must be an installed or a source collection directory.')
    tmp_inst_req = cls(None, None, dir_path, 'dir', None)
    req_version = art_mgr.get_direct_collection_version(tmp_inst_req)
    try:
        req_name = art_mgr.get_direct_collection_fqcn(tmp_inst_req)
    except TypeError as err:
        display.warning(u"Collection at '{path!s}' has a {manifest_json!s} or {galaxy_yml!s} file but it contains invalid metadata.".format(galaxy_yml=to_text(_GALAXY_YAML), manifest_json=to_text(_MANIFEST_JSON), path=to_text(dir_path, errors='surrogate_or_strict')))
        raise ValueError("Collection at '{path!s}' has invalid metadata".format(path=to_text(dir_path, errors='surrogate_or_strict'))) from err
    return cls(req_name, req_version, dir_path, 'dir', None)