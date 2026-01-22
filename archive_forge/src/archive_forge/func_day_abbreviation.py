import sys
from math import trunc
from typing import (
def day_abbreviation(self, day: int) -> str:
    """Returns the day abbreviation for a specified day of the week.

        :param day: the ``int`` day of the week (1-7).

        """
    return self.day_abbreviations[day]