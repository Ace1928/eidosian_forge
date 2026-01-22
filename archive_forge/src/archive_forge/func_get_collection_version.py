from __future__ import annotations
import datetime
import os
import re
import sys
from functools import partial
import yaml
from voluptuous import All, Any, MultipleInvalid, PREVENT_EXTRA
from voluptuous import Required, Schema, Invalid
from voluptuous.humanize import humanize_error
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.six import string_types
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.version import SemanticVersion
def get_collection_version():
    """Return current collection version, or None if it is not available"""
    import importlib.util
    collection_detail_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tools', 'collection_detail.py')
    collection_detail_spec = importlib.util.spec_from_file_location('collection_detail', collection_detail_path)
    collection_detail = importlib.util.module_from_spec(collection_detail_spec)
    sys.modules['collection_detail'] = collection_detail
    collection_detail_spec.loader.exec_module(collection_detail)
    try:
        result = collection_detail.read_manifest_json('.') or collection_detail.read_galaxy_yml('.')
        return SemanticVersion(result['version'])
    except Exception:
        return None