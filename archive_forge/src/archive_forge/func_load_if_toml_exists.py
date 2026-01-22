from __future__ import annotations
import os
import threading
from copy import deepcopy
from typing import (
from blinker import Signal
import streamlit as st
import streamlit.watcher.path_watcher
from streamlit import file_util, runtime
from streamlit.logger import get_logger
def load_if_toml_exists(self) -> bool:
    """Load secrets.toml files from disk if they exists. If none exist,
        no exception will be raised. (If a file exists but is malformed,
        an exception *will* be raised.)

        Returns True if a secrets.toml file was successfully parsed, False otherwise.

        Thread-safe.
        """
    try:
        self._parse(print_exceptions=False)
        return True
    except FileNotFoundError:
        return False