import numpy as np
def combine_datetime_timedelta_units(datetime_unit, timedelta_unit):
    """
    Return the unit result of combining *datetime_unit* with *timedelta_unit*
    (e.g. by adding or subtracting).  None is returned if combining
    those units is forbidden.
    """
    dt_unit_code = DATETIME_UNITS[datetime_unit]
    td_unit_code = DATETIME_UNITS[timedelta_unit]
    if dt_unit_code == 14:
        return timedelta_unit
    elif td_unit_code == 14:
        return datetime_unit
    if td_unit_code < 2 and dt_unit_code >= 2:
        return None
    if dt_unit_code > td_unit_code:
        return datetime_unit
    else:
        return timedelta_unit