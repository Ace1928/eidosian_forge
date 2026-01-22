import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
def print_latencies(latencies):
    print(f'Avg: {round(sum(latencies) / len(latencies), 2)} ms')
    print(f'P25: {round(latencies[int(len(latencies) * 0.25)], 2)} ms')
    print(f'P50: {round(latencies[int(len(latencies) * 0.5)], 2)} ms')
    print(f'P95: {round(latencies[int(len(latencies) * 0.95)], 2)} ms')
    print(f'P99: {round(latencies[int(len(latencies) * 0.99)], 2)} ms')