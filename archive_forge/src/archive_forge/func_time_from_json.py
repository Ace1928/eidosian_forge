import re
import traitlets
import datetime as dt
def time_from_json(js, manager):
    """Deserialize a Python time object from json."""
    if js is None:
        return None
    else:
        return dt.time(js['hours'], js['minutes'], js['seconds'], js['milliseconds'] * 1000)