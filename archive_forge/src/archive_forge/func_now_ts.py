import ipaddress
import time
from datetime import datetime
from enum import Enum
def now_ts():
    return datetime.now().strftime('%Y%m%d-%H%M%S')