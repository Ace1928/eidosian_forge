from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def share_run(self, run_id: ID_TYPE, *, share_id: Optional[ID_TYPE]=None) -> str:
    """Get a share link for a run."""
    run_id_ = _as_uuid(run_id, 'run_id')
    data = {'run_id': str(run_id_), 'share_token': share_id or str(uuid.uuid4())}
    response = self.session.put(f'{self.api_url}/runs/{run_id_}/share', headers=self._headers, json=data)
    ls_utils.raise_for_status_with_text(response)
    share_token = response.json()['share_token']
    return f'{self._host_url}/public/{share_token}/r'