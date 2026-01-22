import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
def on_renamed(self, new_path: PathStr, new_name: LogicalPath) -> None:
    self.file_path = new_path
    self.save_name = new_name
    self.on_modified()