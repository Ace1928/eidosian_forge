import re
import sys
from typing import List, Optional, Union
def is_next_line_obviously_invalid_request_line(self) -> bool:
    try:
        return self._data[0] < 33
    except IndexError:
        return False