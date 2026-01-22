import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_read_text_does_not_keep_open(self):
    resources.files('data01').joinpath('utf-8.file').read_text(encoding='utf-8')