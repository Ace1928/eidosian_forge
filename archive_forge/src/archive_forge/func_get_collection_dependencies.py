from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
def get_collection_dependencies(self, collection_candidate):
    """Retrieve collection dependencies of a given candidate."""
    if collection_candidate.is_concrete_artifact:
        return self._concrete_art_mgr.get_direct_collection_dependencies(collection_candidate)
    return self.get_collection_version_metadata(collection_candidate).dependencies