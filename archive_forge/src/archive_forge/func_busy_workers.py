import asyncio
import base64
import copy
import logging
import os
import socket
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union
import click
import yaml
import wandb
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.sweeps import SchedulerError
from wandb.sdk.launch.sweeps.utils import (
from wandb.sdk.launch.utils import (
from wandb.sdk.lib.runid import generate_id
@property
def busy_workers(self) -> Dict[int, _Worker]:
    """Returns dict of id:worker already assigned to a launch run.

        runs should always have a worker_id, but are created before
        workers are assigned to the run
        """
    busy_workers = {}
    for _, r in self._yield_runs():
        busy_workers[r.worker_id] = self._workers[r.worker_id]
    return busy_workers