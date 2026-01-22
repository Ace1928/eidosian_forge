import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
def write_neuron_monitor_config(self) -> None:
    """Write neuron monitor config file."""
    pathlib.Path(self.neuron_monitor_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(self.neuron_monitor_config_path, 'w') as f:
        json.dump(NEURON_MONITOR_DEFAULT_CONFIG, f, indent=4)