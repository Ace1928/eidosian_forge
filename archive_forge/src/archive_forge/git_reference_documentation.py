import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union
from wandb.sdk.launch.errors import LaunchError
Fetch the repo into dst_dir and refine githubref based on what we learn.