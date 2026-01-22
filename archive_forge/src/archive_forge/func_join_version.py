import re
def join_version(version_tuple):
    """Return a string representation of version from given VERSION tuple"""
    version = '%s.%s.%s' % version_tuple[:3]
    if version_tuple[3] != 'final':
        version += '-%s' % version_tuple[3]
    return version