from __future__ import annotations
import asyncio
import csv
import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from jupyter_server.base.handlers import APIHandler
from tornado import web
from traitlets import List, Unicode
from traitlets.config import LoggingConfigurable
from .config import get_federated_extensions
def report_json(self, bundles: dict[str, Any]) -> str:
    """create a JSON report
        TODO: SPDX
        """
    return json.dumps({'bundles': bundles}, indent=2, sort_keys=True)