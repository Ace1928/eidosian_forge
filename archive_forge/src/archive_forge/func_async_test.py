from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def async_test(self, fn):
    from sqlalchemy.testing import asyncio

    @_pytest_fn_decorator
    def decorate(fn, *args, **kwargs):
        asyncio._run_coroutine_function(fn, *args, **kwargs)
    return decorate(fn)