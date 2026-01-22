from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def get_tab(self, name) -> Union[CardTab, None]:
    """
        Returns an existing tab with the specified name. Returns None if not found.

        Args:
            name: A string representing the name of the tab.

        Returns:
            An existing tab with the specified name. If not found, returns None.
        """
    for key, tab in self._tabs:
        if key == name:
            return tab
    return None