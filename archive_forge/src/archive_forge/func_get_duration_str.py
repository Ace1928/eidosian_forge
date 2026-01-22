import abc
import datetime as dt
import textwrap
from osc_lib.command import command
def get_duration_str(start_dt_str, end_dt_str):
    """Builds a human friendly duration string.

    :param start_dt_str: Start date time as an ISO string.
    :param end_dt_str: End date time as an ISO string. If empty, duration is
        calculated from the current time.
    :return: Duration(delta) string.
    """
    if not start_dt_str:
        return ''
    start_dt = dt.datetime.strptime(start_dt_str, '%Y-%m-%d %H:%M:%S')
    if end_dt_str:
        end_dt = dt.datetime.strptime(end_dt_str, '%Y-%m-%d %H:%M:%S')
        return str(end_dt - start_dt)
    delta_from_now = dt.datetime.utcnow() - start_dt
    if delta_from_now < dt.timedelta(seconds=2):
        return '...'
    delta = delta_from_now - dt.timedelta(microseconds=delta_from_now.microseconds)
    return '{}...'.format(delta)