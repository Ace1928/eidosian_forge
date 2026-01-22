from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.urls import open_url
import contextlib
import os
import subprocess
import sys
import typing as t
from dataclasses import dataclass, fields as dc_fields
from functools import partial
from urllib.error import HTTPError, URLError
def get_signature_from_source(source, display=None):
    if display is not None:
        display.vvvv(f'Using signature at {source}')
    try:
        with open_url(source, http_agent=user_agent(), validate_certs=True, follow_redirects='safe') as resp:
            signature = resp.read()
    except (HTTPError, URLError) as e:
        raise AnsibleError(f"Failed to get signature for collection verification from '{source}': {e}") from e
    return signature