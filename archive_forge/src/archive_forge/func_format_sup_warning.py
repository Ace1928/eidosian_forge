import re
import html
from paste.util import PySourceColor
def format_sup_warning(self, warning):
    return 'Warning: %s' % self.quote(warning)