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
def list_runs_from_annotation_queue(self, queue_id: ID_TYPE, *, limit: Optional[int]=None) -> Iterator[ls_schemas.RunWithAnnotationQueueInfo]:
    """List runs from an annotation queue with the specified queue ID.

        Args:
            queue_id (ID_TYPE): The ID of the annotation queue.

        Yields:
            ls_schemas.RunWithAnnotationQueueInfo: An iterator of runs from the
                annotation queue.
        """
    path = f'/annotation-queues/{_as_uuid(queue_id, 'queue_id')}/runs'
    limit_ = min(limit, 100) if limit is not None else 100
    for i, run in enumerate(self._get_paginated_list(path, params={'headers': self._headers, 'limit': limit_})):
        yield ls_schemas.RunWithAnnotationQueueInfo(**run)
        if limit is not None and i + 1 >= limit:
            break