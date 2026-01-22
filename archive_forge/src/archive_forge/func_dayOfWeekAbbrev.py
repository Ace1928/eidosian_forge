import re, time, datetime
from .utils import isStr
def dayOfWeekAbbrev(self):
    """return day of week abbreviation for current date: Mon, Tue, etc."""
    return self.__day_of_week_name__[self.dayOfWeek()][:3]