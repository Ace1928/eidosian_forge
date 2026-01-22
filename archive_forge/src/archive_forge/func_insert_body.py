import os
import re
from paste.fileapp import FileApp
from paste.response import header_value, remove_header
def insert_body(body, text):
    end_body = re.search('</body>', body, re.I)
    if end_body:
        return body[:end_body.start()] + text + body[end_body.end():]
    else:
        return body + text