import sys
from math import trunc
from typing import (
def month_abbreviation(self, month: int) -> str:
    """Returns the month abbreviation for a specified month of the year.

        :param month: the ``int`` month of the year (1-12).

        """
    return self.month_abbreviations[month]