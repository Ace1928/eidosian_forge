import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_duration(cls, PnY=None, PnM=None, PnW=None, PnD=None, TnH=None, TnM=None, TnS=None):
    return DurationTuple(PnY, PnM, PnW, PnD, TnH, TnM, TnS)