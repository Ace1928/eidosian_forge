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
def before_test(test, test_module_name, test_class, test_name):
    name = getattr(test_class, '_sa_orig_cls_name', test_class.__name__)
    id_ = '%s.%s.%s' % (test_module_name, name, test_name)
    profiling._start_current_test(id_)