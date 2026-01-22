import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
def dtime(cls, dt: str=None, prefer: str='past'):
    return LazyDate.dtime(dt, prefer)