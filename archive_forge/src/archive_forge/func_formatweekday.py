import sys
import datetime
import locale as _locale
from itertools import repeat
def formatweekday(self, day):
    with different_locale(self.locale):
        return super().formatweekday(day)