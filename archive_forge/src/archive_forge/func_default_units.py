from matplotlib import cbook, units
import matplotlib.dates as date_ticker
@staticmethod
def default_units(value, axis):
    if cbook.is_scalar_or_string(value):
        return value.frame()
    else:
        return EpochConverter.default_units(value[0], axis)