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
def filter_instance(instance: GCPTPUNode) -> bool:
    if instance.is_terminated():
        return False
    labels = instance.get_labels()
    if label_filters:
        for key, value in label_filters.items():
            if key not in labels:
                return False
            if value != labels[key]:
                return False
    return True