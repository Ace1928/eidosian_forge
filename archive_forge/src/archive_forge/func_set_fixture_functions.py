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
def set_fixture_functions(fixture_fn_class):
    global _fixture_fn_class
    _fixture_fn_class = fixture_fn_class