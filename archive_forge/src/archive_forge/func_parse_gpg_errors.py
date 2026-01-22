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
def parse_gpg_errors(status_out):
    for line in status_out.splitlines():
        if not line:
            continue
        try:
            _dummy, status, remainder = line.split(maxsplit=2)
        except ValueError:
            _dummy, status = line.split(maxsplit=1)
            remainder = None
        try:
            cls = GPG_ERROR_MAP[status]
        except KeyError:
            continue
        fields = [status]
        if remainder:
            fields.extend(remainder.split(None, len(dc_fields(cls)) - 2))
        yield cls(*fields)