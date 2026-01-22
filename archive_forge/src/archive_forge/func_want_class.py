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
def want_class(name, cls):
    if not issubclass(cls, fixtures.TestBase):
        return False
    elif name.startswith('_'):
        return False
    else:
        return True