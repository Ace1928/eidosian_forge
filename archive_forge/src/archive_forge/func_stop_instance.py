from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
def stop_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
    operation = self.resource.projects().locations().nodes().stop(name=node_id).execute()
    if wait_for_operation:
        result = self.wait_for_operation(operation, max_polls=MAX_POLLS)
    else:
        result = operation
    return result