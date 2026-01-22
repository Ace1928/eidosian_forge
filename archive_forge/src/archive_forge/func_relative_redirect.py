from paste.wsgilib import catch_errors_app
from paste.response import has_header, header_value, replace_header
from paste.request import resolve_relative_url
from paste.util.quoting import strip_html, html_quote, no_quote, comment_quote
def relative_redirect(cls, dest_uri, environ, detail=None, headers=None, comment=None):
    """
        Create a redirect object with the dest_uri, which may be relative,
        considering it relative to the uri implied by the given environ.
        """
    location = resolve_relative_url(dest_uri, environ)
    headers = headers or []
    headers.append(('Location', location))
    return cls(detail=detail, headers=headers, comment=comment)