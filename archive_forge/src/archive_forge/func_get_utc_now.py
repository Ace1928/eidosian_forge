import datetime
import pytz
import six
def get_utc_now(timezone=None):
    """Return the current UTC time.

    :param timezone: an optional timezone param to offset time to.
    """
    utc_datetime = pytz.utc.localize(datetime.datetime.utcnow())
    if timezone is not None:
        try:
            utc_datetime = utc_datetime.astimezone(pytz.timezone(timezone))
        except Exception:
            utc_datetime.strftime(TIME_FORMAT)
    return utc_datetime.strftime(TIME_FORMAT)