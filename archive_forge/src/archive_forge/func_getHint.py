import calendar
from typing import Any, Optional, Tuple
def getHint(self, name, default=None):
    return self.hints.get(name, default)