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
def list_shared_runs(self, share_token: ID_TYPE, run_ids: Optional[List[str]]=None) -> List[ls_schemas.Run]:
    """Get shared runs."""
    params = {'id': run_ids, 'share_token': str(share_token)}
    response = self.session.get(f'{self.api_url}/public/{_as_uuid(share_token, 'share_token')}/runs', headers=self._headers, params=params)
    ls_utils.raise_for_status_with_text(response)
    return [ls_schemas.Run(**run, _host_url=self._host_url) for run in response.json()]