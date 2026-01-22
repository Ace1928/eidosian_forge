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
def list_annotation_queues(self, *, queue_ids: Optional[List[ID_TYPE]]=None, name: Optional[str]=None, name_contains: Optional[str]=None, limit: Optional[int]=None) -> Iterator[ls_schemas.AnnotationQueue]:
    """List the annotation queues on the LangSmith API.

        Args:
            queue_ids : List[str or UUID] or None, default=None
                The IDs of the queues to filter by.
            name : str or None, default=None
                The name of the queue to filter by.
            name_contains : str or None, default=None
                The substring that the queue name should contain.
            limit : int or None, default=None

        Yields:
            AnnotationQueue
                The annotation queues.
        """
    params: dict = {'ids': [_as_uuid(id_, f'queue_ids[{i}]') for i, id_ in enumerate(queue_ids)] if queue_ids is not None else None, 'name': name, 'name_contains': name_contains, 'limit': min(limit, 100) if limit is not None else 100}
    for i, queue in enumerate(self._get_paginated_list('/annotation-queues', params=params)):
        yield ls_schemas.AnnotationQueue(**queue, _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())
        if limit is not None and i + 1 >= limit:
            break