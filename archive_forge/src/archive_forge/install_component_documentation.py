from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from rich.markup import escape
from typer import Argument, Option
from typing_extensions import Annotated
from gradio.cli.commands.display import LivePanelDisplay
from gradio.utils import set_directory
Get the path to an executable, either from the provided path or from the PATH environment variable.

    The value of executable_path takes precedence in case the value in PATH is incorrect.
    This should give more control to the developer in case their envrinment is not set up correctly.

    If check_3 is True, we append 3 to the executable name to give python3 priority over python (same for pip).
    