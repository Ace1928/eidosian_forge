import configparser
import json
import re
import shutil
import subprocess
import sys
from typing import Optional
from pathlib import Path
Update an extension to the current JupyterLab

    target: str
        Path to the extension directory containing the extension
    vcs_ref: str [default: None]
        Template vcs_ref to checkout
    interactive: bool [default: true]
        Whether to ask before overwriting content

    