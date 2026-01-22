from fnmatch import fnmatch
import os
from pathlib import Path
import re
import subprocess
import pytest
import cartopy

        Return a list of all the files under git.

        .. note::

            This function raises a ValueError if the repo root does
            not have a ".git" folder. If git is not installed on the system,
            or cannot be found by subprocess, an IOError may also be raised.

        