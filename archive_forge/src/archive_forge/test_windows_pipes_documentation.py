from __future__ import annotations
import sys
from typing import TYPE_CHECKING
import pytest
from .. import _core
from ..testing import check_one_way_stream, wait_all_tasks_blocked
Makes a new pair of pipes.