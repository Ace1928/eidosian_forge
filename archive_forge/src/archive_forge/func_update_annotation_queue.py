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
def update_annotation_queue(self, queue_id: ID_TYPE, *, name: str, description: Optional[str]=None) -> None:
    """Update an annotation queue with the specified queue_id.

        Args:
            queue_id (ID_TYPE): The ID of the annotation queue to update.
            name (str): The new name for the annotation queue.
            description (Optional[str], optional): The new description for the
                annotation queue. Defaults to None.
        """
    response = self.request_with_retries('PATCH', f'/annotation-queues/{_as_uuid(queue_id, 'queue_id')}', json={'name': name, 'description': description})
    ls_utils.raise_for_status_with_text(response)