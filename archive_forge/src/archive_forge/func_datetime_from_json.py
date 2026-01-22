import re
import traitlets
import datetime as dt
def datetime_from_json(js, manager):
    """Deserialize a Python datetime object from json."""
    if js is None:
        return None
    else:
        try:
            return dt.datetime(js['year'], js['month'] + 1, js['date'], js['hours'], js['minutes'], js['seconds'], js['milliseconds'] * 1000, dt.timezone.utc).astimezone()
        except (ValueError, OSError):
            return dt.datetime(js['year'], js['month'] + 1, js['date'], js['hours'], js['minutes'], js['seconds'], js['milliseconds'] * 1000, dt.timezone.utc)