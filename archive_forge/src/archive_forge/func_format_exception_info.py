import re
import html
from paste.util import PySourceColor
def format_exception_info(self, etype, evalue):
    return self.emphasize('%s: %s' % (self.quote(etype), self.quote(evalue)))