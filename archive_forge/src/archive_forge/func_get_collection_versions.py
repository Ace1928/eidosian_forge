from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
def get_collection_versions(self, requirement):
    """Get a set of unique versions for FQCN on Galaxy servers."""
    if requirement.is_concrete_artifact:
        return {(self._concrete_art_mgr.get_direct_collection_version(requirement), requirement.src)}
    api_lookup_order = (requirement.src,) if isinstance(requirement.src, GalaxyAPI) else self._apis
    return set(((version, api) for api, version in self._get_collection_versions(requirement)))