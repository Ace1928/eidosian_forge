import os
import sys
from typing import IO, TYPE_CHECKING, Optional
from wandb.errors import CommError
Fallback to the file object for attrs not defined here.