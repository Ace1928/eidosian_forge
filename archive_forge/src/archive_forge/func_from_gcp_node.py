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
@staticmethod
def from_gcp_node(node: 'GCPNode'):
    """Return GCPNodeType based on ``node``'s class"""
    if isinstance(node, GCPTPUNode):
        return GCPNodeType.TPU
    if isinstance(node, GCPComputeNode):
        return GCPNodeType.COMPUTE
    raise TypeError(f'Wrong GCPNode type {type(node)}.')