from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
True if all trials are terminated.