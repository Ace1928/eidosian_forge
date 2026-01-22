import re, time, datetime
from .utils import isStr
def dayOfWeekName(self):
    """return day of week name for current date: Monday, Tuesday, etc."""
    return self.__day_of_week_name__[self.dayOfWeek()]