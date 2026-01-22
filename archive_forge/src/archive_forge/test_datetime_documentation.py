import copy
import pytest  # NOQA
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA

http://yaml.org/type/timestamp.html specifies the regexp to use
for datetime.date and datetime.datetime construction. Date is simple
but datetime can have 'T' or 't' as well as 'Z' or a timezone offset (in
hours and minutes). This information was originally used to create
a UTC datetime and then discarded

examples from the above:

canonical:        2001-12-15T02:59:43.1Z
valid iso8601:    2001-12-14t21:59:43.10-05:00
space separated:  2001-12-14 21:59:43.10 -5
no time zone (Z): 2001-12-15 2:59:43.10
date (00:00:00Z): 2002-12-14

Please note that a fraction can only be included if not equal to 0

