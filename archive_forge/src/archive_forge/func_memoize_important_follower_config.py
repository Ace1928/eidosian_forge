from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
def memoize_important_follower_config(dict_):
    """Store important configuration we will need to send to a follower.

    This invokes in the parent process after normal config is set up.

    Hook is currently not used.

    """