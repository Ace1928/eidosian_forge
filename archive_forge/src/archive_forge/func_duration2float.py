from matplotlib import cbook, units
import matplotlib.dates as date_ticker
@staticmethod
def duration2float(value):
    """
        Convert a Duration value to a float suitable for plotting as a python
        datetime object.

        = INPUT VARIABLES
        - value    A Duration or list of Durations that need to be converted.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
    return value.seconds() / 86400.0