from __future__ import annotations
from typing import Final
from streamlit import util
from streamlit.logger import get_logger
Get the *local* IP address of the current machine.

    From: https://stackoverflow.com/a/28950776

    Returns
    -------
    string
        The local IPv4 address of the current machine.

    