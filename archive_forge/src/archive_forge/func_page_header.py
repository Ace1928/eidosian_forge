import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
def page_header(self, request, title):
    """Render the page header"""
    request['body'].append('<html>\n  <head><title>%s</title></head>\n  <style type="text/css">\n      * {\n        font-family: verdana,sans-serif;\n      }\n      body {\n        width: 50em;\n        margin: 1em;\n      }\n      div {\n        padding: .5em;\n      }\n      table {\n        margin: none;\n        padding: none;\n      }\n      .alert {\n        border: 1px solid #e7dc2b;\n        background: #fff888;\n      }\n      .error {\n        border: 1px solid #ff0000;\n        background: #ffaaaa;\n      }\n      #verify-form {\n        border: 1px solid #777777;\n        background: #dddddd;\n        margin-top: 1em;\n        padding-bottom: 0em;\n      }\n  </style>\n  <body>\n    <h1>%s</h1>\n    <p>\n      This example consumer uses the <a\n      href="http://openid.schtuff.com/">Python OpenID</a> library. It\n      just verifies that the URL that you enter is your identity URL.\n    </p>\n' % (title, title))