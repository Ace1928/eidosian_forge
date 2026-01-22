from __future__ import annotations
import glob
import json
import os
from typing import Any
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
from ...utils import url_path_join, url_unescape
def is_kernelspec_model(spec_dict):
    """Returns True if spec_dict is already in proper form.  This will occur when using a gateway."""
    return isinstance(spec_dict, dict) and 'name' in spec_dict and ('spec' in spec_dict) and ('resources' in spec_dict)