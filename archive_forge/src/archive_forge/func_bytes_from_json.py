import re
import traitlets
import datetime as dt
def bytes_from_json(js, obj):
    return None if js is None else js.tobytes()