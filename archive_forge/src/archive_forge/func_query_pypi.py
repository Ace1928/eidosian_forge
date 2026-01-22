import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
def query_pypi(package, include_prereleases):
    """Return information about the current version of package."""
    try:
        response = requests.get(f'https://pypi.org/pypi/{package}/json', timeout=1)
    except requests.exceptions.RequestException:
        return {'success': False}
    if response.status_code != 200:
        return {'success': False}
    data = response.json()
    versions = list(data['releases'].keys())
    versions.sort(key=parse_version, reverse=True)
    version = versions[0]
    for tmp_version in versions:
        if include_prereleases or standard_release(tmp_version):
            version = tmp_version
            break
    upload_time = None
    for file_info in data['releases'][version]:
        if file_info['upload_time']:
            upload_time = file_info['upload_time']
            break
    return {'success': True, 'data': {'upload_time': upload_time, 'version': version}}