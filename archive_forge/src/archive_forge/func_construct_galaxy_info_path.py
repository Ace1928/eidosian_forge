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
def construct_galaxy_info_path(self, b_collection_path):
    if not self.may_have_offline_galaxy_info and (not self.type == 'galaxy'):
        raise TypeError('Only installed collections from a Galaxy server have offline Galaxy info')
    b_src = to_bytes(b_collection_path, errors='surrogate_or_strict')
    b_path_parts = b_src.split(to_bytes(os.path.sep))[0:-2]
    b_metadata_dir = to_bytes(os.path.sep).join(b_path_parts)
    b_dir_name = to_bytes(f'{self.namespace}.{self.name}-{self.ver}.info', errors='surrogate_or_strict')
    return os.path.join(b_metadata_dir, b_dir_name, _SOURCE_METADATA_FILE)