import datetime
import calendar
import operator
from math import copysign
from six import integer_types
from warnings import warn
from ._common import weekday

        Return a version of this object represented entirely using integer
        values for the relative attributes.

        >>> relativedelta(days=1.5, hours=2).normalized()
        relativedelta(days=+1, hours=+14)

        :return:
            Returns a :class:`dateutil.relativedelta.relativedelta` object.
        