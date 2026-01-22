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
def list_shared_projects(self, *, dataset_share_token: str, project_ids: Optional[List[ID_TYPE]]=None, name: Optional[str]=None, name_contains: Optional[str]=None, limit: Optional[int]=None) -> Iterator[ls_schemas.TracerSessionResult]:
    """List shared projects.

        Args:
            dataset_share_token : str
                The share token of the dataset.
            project_ids : List[ID_TYPE], optional
                List of project IDs to filter the results, by default None.
            name : str, optional
                Name of the project to filter the results, by default None.
            name_contains : str, optional
                Substring to search for in project names, by default None.
            limit : int, optional

        Yields:
            TracerSessionResult: The shared projects.
        """
    params = {'id': project_ids, 'name': name, 'name_contains': name_contains}
    share_token = _as_uuid(dataset_share_token, 'dataset_share_token')
    for i, project in enumerate(self._get_paginated_list(f'/public/{share_token}/datasets/sessions', params=params)):
        yield ls_schemas.TracerSessionResult(**project, _host_url=self._host_url)
        if limit is not None and i + 1 >= limit:
            break