from __future__ import annotations
import json
import os
import re
import sys
import yaml
def read_manifest_json(collection_path):
    """Return collection information from the MANIFEST.json file."""
    manifest_path = os.path.join(collection_path, 'MANIFEST.json')
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
        collection_info = manifest.get('collection_info') or {}
        result = dict(version=collection_info.get('version'))
        validate_version(result['version'])
    except Exception as ex:
        raise Exception('{0}: {1}'.format(os.path.basename(manifest_path), ex)) from None
    return result