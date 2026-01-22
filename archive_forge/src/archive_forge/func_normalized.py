import datetime
import calendar
import operator
from math import copysign
from six import integer_types
from warnings import warn
from ._common import weekday
def normalized(self):
    """
        Return a version of this object represented entirely using integer
        values for the relative attributes.

        >>> relativedelta(days=1.5, hours=2).normalized()
        relativedelta(days=+1, hours=+14)

        :return:
            Returns a :class:`dateutil.relativedelta.relativedelta` object.
        """
    days = int(self.days)
    hours_f = round(self.hours + 24 * (self.days - days), 11)
    hours = int(hours_f)
    minutes_f = round(self.minutes + 60 * (hours_f - hours), 10)
    minutes = int(minutes_f)
    seconds_f = round(self.seconds + 60 * (minutes_f - minutes), 8)
    seconds = int(seconds_f)
    microseconds = round(self.microseconds + 1000000.0 * (seconds_f - seconds))
    return self.__class__(years=self.years, months=self.months, days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds, leapdays=self.leapdays, year=self.year, month=self.month, day=self.day, weekday=self.weekday, hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond)