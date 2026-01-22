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
def pytest_testnodedown(self, node, error):
    from sqlalchemy.testing import provision
    from sqlalchemy.testing import asyncio
    asyncio._maybe_async_provisioning(provision.drop_follower_db, node.workerinput['follower_ident'])