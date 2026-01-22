from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def compute_timestamp(t):
    now = datetime.now().replace(microsecond=0)
    if 'computed_time' not in t:
        t['computed_time'] = t.ref_time or now.time()
    if 'abs_date' not in t:
        t['abs_date'] = now
    t['computed_dt'] = t.abs_date.replace(hour=t.computed_time.hour, minute=t.computed_time.minute, second=t.computed_time.second) + (t.time_delta or timedelta(0)) + (t.date_delta or timedelta(0))
    if not t.time_ref_present:
        t['computed_dt'] = t.computed_dt.replace(hour=0, minute=0, second=0)
    t['calculatedTime'] = t.computed_dt
    t['time_offset'] = t.computed_dt - t.relative_to