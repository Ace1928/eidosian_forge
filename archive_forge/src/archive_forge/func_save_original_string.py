from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def save_original_string(s, l, t):
    t['original'] = ' '.join(s.strip().split())
    t['relative_to'] = datetime.now().replace(microsecond=0)