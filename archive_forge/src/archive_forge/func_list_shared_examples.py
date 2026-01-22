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
def list_shared_examples(self, share_token: str, *, example_ids: Optional[List[ID_TYPE]]=None) -> List[ls_schemas.Example]:
    """Get shared examples."""
    params = {}
    if example_ids is not None:
        params['id'] = [str(id) for id in example_ids]
    response = self.session.get(f'{self.api_url}/public/{_as_uuid(share_token, 'share_token')}/examples', headers=self._headers, params=params)
    ls_utils.raise_for_status_with_text(response)
    return [ls_schemas.Example(**dataset, _host_url=self._host_url) for dataset in response.json()]