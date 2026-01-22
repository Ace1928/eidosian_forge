import re
import html
from paste.util import PySourceColor
def format_sup_url(self, url):
    return 'URL: <a href="%s">%s</a>' % (url, url)