import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
def uname_attr(self, attribute: str) -> str:
    """
        Return a single named information item from the uname command
        output data source of the OS distribution.

        For details, see :func:`distro.uname_attr`.
        """
    return self._uname_info.get(attribute, '')