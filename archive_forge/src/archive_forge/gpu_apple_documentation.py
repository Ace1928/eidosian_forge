import json
import logging
import pathlib
import platform
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
Apple GPU stats available on Arm Macs.