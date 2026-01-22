import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def new_sha256(*args):
    raise NotImplementedError('Built-in sha1 implementation not found; cannot use hashlib implementation because it depends on OpenSSL, which may not be linked with this library due to license incompatibilities')