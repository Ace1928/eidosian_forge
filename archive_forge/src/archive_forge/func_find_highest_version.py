import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
import numpy as np
from gym.wrappers import (
from gym.wrappers.compatibility import EnvCompatibility
from gym.wrappers.env_checker import PassiveEnvChecker
from gym import Env, error, logger
def find_highest_version(ns: Optional[str], name: str) -> Optional[int]:
    version: List[int] = [spec_.version for spec_ in registry.values() if spec_.namespace == ns and spec_.name == name and (spec_.version is not None)]
    return max(version, default=None)