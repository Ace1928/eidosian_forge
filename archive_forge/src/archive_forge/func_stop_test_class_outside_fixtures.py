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
def stop_test_class_outside_fixtures(cls):
    engines.testing_reaper.stop_test_class_outside_fixtures()
    provision.stop_test_class_outside_fixtures(config, config.db, cls)
    try:
        if not options.low_connections:
            assertions.global_cleanup_assertions()
    finally:
        _restore_engine()