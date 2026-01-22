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
@abc.abstractmethod
def combinations(self, *args, **kw):
    raise NotImplementedError()