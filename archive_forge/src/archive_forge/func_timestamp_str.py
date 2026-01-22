from collections import abc, defaultdict
import datetime
from itertools import cycle
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import google.protobuf.json_format as json_format
import cirq
from cirq_google.api import v2
def timestamp_str(self, tz: Optional[datetime.tzinfo]=None, timespec: str='auto') -> str:
    """Return a string for the calibration timestamp.

        Args:
            tz: The timezone for the string. If None, the method uses the
                platform's local timezone.
            timespec: See datetime.isoformat for valid values.

        Returns:
            The string in ISO 8601 format YYYY-MM-DDTHH:MM:SS.ffffff.
        """
    dt = datetime.datetime.fromtimestamp(self.timestamp / 1000, tz)
    dt += datetime.timedelta(microseconds=self.timestamp % 1000000)
    return dt.isoformat(sep=' ', timespec=timespec)