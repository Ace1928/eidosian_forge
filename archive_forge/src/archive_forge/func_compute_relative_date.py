from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def compute_relative_date(t):
    now = datetime.now().replace(microsecond=0)
    if 'ref_day' in t:
        t['computed_date'] = t.ref_day
    else:
        t['computed_date'] = now.date()
    day_diff = t.dir * t.qty * {'week': 7, 'day': 1}[t.units]
    t['date_delta'] = timedelta(days=day_diff)