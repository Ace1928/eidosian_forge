import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
def page_footer(self, request, form_contents):
    """Render the page footer"""
    if not form_contents:
        form_contents = ''
    request['body'].append('    <div id="verify-form">\n      <form method="get" action=%s>\n        Identity&nbsp;URL:\n        <input type="text" name="openid_url" value=%s />\n        <input type="submit" value="Verify" />\n      </form>\n    </div>\n  </body>\n</html>\n' % (quoteattr(self.build_url(request, 'verify')), quoteattr(form_contents)))