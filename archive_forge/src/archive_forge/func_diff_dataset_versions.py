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
def diff_dataset_versions(self, dataset_id: Optional[ID_TYPE]=None, *, dataset_name: Optional[str]=None, from_version: Union[str, datetime.datetime], to_version: Union[str, datetime.datetime]) -> ls_schemas.DatasetDiffInfo:
    """Get the difference between two versions of a dataset.

        Parameters
        ----------
        dataset_id : str or None, default=None
            The ID of the dataset.
        dataset_name : str or None, default=None
            The name of the dataset.
        from_version : str or datetime.datetime
            The starting version for the diff.
        to_version : str or datetime.datetime
            The ending version for the diff.

        Returns:
        -------
        DatasetDiffInfo
            The difference between the two versions of the dataset.

        Examples:
        --------
        ..code-block:: python

            # Get the difference between two tagged versions of a dataset
            from_version = "prod"
            to_version = "dev"
            diff = client.diff_dataset_versions(
                dataset_name="my-dataset",
                from_version=from_version,
                to_version=to_version,
            )
            print(diff)

            # Get the difference between two timestamped versions of a dataset

            from_version = datetime.datetime(2024, 1, 1)
            to_version = datetime.datetime(2024, 2, 1)
            diff = client.diff_dataset_versions(
                dataset_name="my-dataset",
                from_version=from_version,
                to_version=to_version,
            )
            print(diff)
        """
    if dataset_id is None:
        if dataset_name is None:
            raise ValueError('Must provide either dataset name or ID')
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    dsid = _as_uuid(dataset_id, 'dataset_id')
    response = self.session.get(f'{self.api_url}/datasets/{dsid}/versions/diff', headers=self._headers, params={'from_version': from_version.isoformat() if isinstance(from_version, datetime.datetime) else from_version, 'to_version': to_version.isoformat() if isinstance(to_version, datetime.datetime) else to_version})
    ls_utils.raise_for_status_with_text(response)
    return ls_schemas.DatasetDiffInfo(**response.json())