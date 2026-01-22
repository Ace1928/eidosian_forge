import logging
import os
import sys
from pathlib import Path
from typing import List
from . import MAJOR_VERSION, run_command

    Wrapper class around the nuitka executable, enabling its usage through python code
    