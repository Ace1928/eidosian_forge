import datetime
import os
import time
from contextlib import contextmanager
import pytest
from dateutil import tz

    Switch to a locally-known timezone specified by `tzname`.
    On exit, restore the previous timezone.
    If `tzname` is `None`, do nothing.
    