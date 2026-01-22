import io
import os.path
def extract_local_path(uri_as_string):
    if uri_as_string.startswith('file://'):
        local_path = uri_as_string.replace('file://', '', 1)
    else:
        local_path = uri_as_string
    return os.path.expanduser(local_path)