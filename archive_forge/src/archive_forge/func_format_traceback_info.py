import re
import html
from paste.util import PySourceColor
def format_traceback_info(self, info):
    return '<pre>%s</pre>' % self.quote(info)