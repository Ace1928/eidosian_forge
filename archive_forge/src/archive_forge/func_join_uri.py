from .schema import rest_translation
def join_uri(uri, *segments):
    part1 = [seg.lstrip('/') for seg in segments]
    return '/'.join(uri.split('/') + part1).rstrip('/')