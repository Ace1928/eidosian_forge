from paste.wsgilib import catch_errors_app
from paste.response import has_header, header_value, replace_header
from paste.request import resolve_relative_url
from paste.util.quoting import strip_html, html_quote, no_quote, comment_quote
def prepare_content(self, environ):
    if self.headers:
        headers = list(self.headers)
    else:
        headers = []
    if 'html' in environ.get('HTTP_ACCEPT', '') or '*/*' in environ.get('HTTP_ACCEPT', ''):
        replace_header(headers, 'content-type', 'text/html')
        content = self.html(environ)
    else:
        replace_header(headers, 'content-type', 'text/plain')
        content = self.plain(environ)
    if isinstance(content, str):
        content = content.encode('utf8')
        cur_content_type = header_value(headers, 'content-type') or 'text/html'
        replace_header(headers, 'content-type', cur_content_type + '; charset=utf8')
    return (headers, content)