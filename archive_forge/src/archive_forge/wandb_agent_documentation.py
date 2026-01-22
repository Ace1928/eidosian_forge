import logging
import multiprocessing
import os
import platform
import queue
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import yaml
import wandb
from wandb import util, wandb_lib, wandb_sdk
from wandb.agents.pyagent import pyagent
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
Determine if the process is flapping.

        Flapping occurs if the agents receives FLAPPING_MAX_FAILURES non-0 exit codes in
        the first FLAPPING_MAX_SECONDS.
        