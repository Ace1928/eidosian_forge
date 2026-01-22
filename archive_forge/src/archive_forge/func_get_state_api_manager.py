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
def get_state_api_manager(gcs_address: str) -> StateAPIManager:
    gcs_aio_client = GcsAioClient(address=gcs_address)
    gcs_channel = GcsChannel(gcs_address=gcs_address, aio=True)
    gcs_channel.connect()
    state_api_data_source_client = StateDataSourceClient(gcs_channel.channel(), gcs_aio_client)
    return StateAPIManager(state_api_data_source_client)