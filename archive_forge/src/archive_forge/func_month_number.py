import sys
from math import trunc
from typing import (
def month_number(self, name: str) -> Optional[int]:
    """Returns the month number for a month specified by name or abbreviation.

        :param name: the month name or abbreviation.

        """
    if self._month_name_to_ordinal is None:
        self._month_name_to_ordinal = self._name_to_ordinal(self.month_names)
        self._month_name_to_ordinal.update(self._name_to_ordinal(self.month_abbreviations))
    return self._month_name_to_ordinal.get(name)