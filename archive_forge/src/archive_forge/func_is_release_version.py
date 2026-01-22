import re
def is_release_version():
    return bool(re.match('^\\d+\\.\\d+\\.\\d+$', VERSION))