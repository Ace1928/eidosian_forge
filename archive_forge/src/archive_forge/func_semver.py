import logging
import re
def semver(version, loose):
    if isinstance(version, SemVer):
        if version.loose == loose:
            return version
        else:
            version = version.version
    elif not isinstance(version, string_type):
        raise ValueError(f'Invalid Version: {version}')
    '\n    if (!(this instanceof SemVer))\n       return new SemVer(version, loose);\n    '
    return SemVer(version, loose)