import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def placement_group_assert_no_leak(pgs_created):
    for pg in pgs_created:
        ray.util.remove_placement_group(pg)

    def wait_for_pg_removed():
        for pg_entry in ray.util.placement_group_table().values():
            if pg_entry['state'] != 'REMOVED':
                return False
        return True
    wait_for_condition(wait_for_pg_removed)
    cluster_resources = ray.cluster_resources()
    cluster_resources.pop('memory')
    cluster_resources.pop('object_store_memory')

    def wait_for_resource_recovered():
        for resource, val in ray.available_resources().items():
            if resource in cluster_resources and cluster_resources[resource] != val:
                return False
            if '_group_' in resource:
                return False
        return True
    wait_for_condition(wait_for_resource_recovered)