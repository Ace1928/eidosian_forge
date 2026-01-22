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
def read_dataset_shared_schema(self, dataset_id: Optional[ID_TYPE]=None, *, dataset_name: Optional[str]=None) -> ls_schemas.DatasetShareSchema:
    """Retrieve the shared schema of a dataset.

        Args:
            dataset_id (Optional[ID_TYPE]): The ID of the dataset.
                Either `dataset_id` or `dataset_name` must be given.
            dataset_name (Optional[str]): The name of the dataset.
                Either `dataset_id` or `dataset_name` must be given.

        Returns:
            ls_schemas.DatasetShareSchema: The shared schema of the dataset.

        Raises:
            ValueError: If neither `dataset_id` nor `dataset_name` is given.
        """
    if dataset_id is None and dataset_name is None:
        raise ValueError('Either dataset_id or dataset_name must be given')
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    response = self.session.get(f'{self.api_url}/datasets/{_as_uuid(dataset_id, 'dataset_id')}/share', headers=self._headers)
    ls_utils.raise_for_status_with_text(response)
    d = response.json()
    return cast(ls_schemas.DatasetShareSchema, {**d, 'url': f'{self._host_url}/public/{_as_uuid(d['share_token'], 'response.share_token')}/d'})