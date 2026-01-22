import re
def find_uris_in_text(text):
    """Scan a block of text for URIs, and yield the ones found."""
    for match in possible_uri_pat.finditer(text):
        uri_string = match.group()
        uri_string = uri_trailers_pat.sub('', uri_string)
        try:
            uri = URI(uri_string)
        except InvalidURIError:
            continue
        yield uri