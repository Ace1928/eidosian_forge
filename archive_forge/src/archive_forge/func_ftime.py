import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
def ftime(cls, t: time.time=None, short: bool=True):
    return LazyTime(t, short=short)