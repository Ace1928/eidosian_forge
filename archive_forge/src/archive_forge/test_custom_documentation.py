import unittest
import contextlib
import pathlib
import importlib_resources as resources
from .. import abc
from ..abc import TraversableResources, ResourceReader
from . import util
from .compat.py39 import os_helper

    Magically returns the resources at path.
    