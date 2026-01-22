import optparse
import sys
import re
import os
from .diff import htmldiff
def split_body(html):
    pre = post = ''
    match = body_start_re.search(html)
    if match:
        pre = html[:match.end()]
        html = html[match.end():]
    match = body_end_re.search(html)
    if match:
        post = html[match.start():]
        html = html[:match.start()]
    return (pre, html, post)