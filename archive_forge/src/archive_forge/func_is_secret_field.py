from __future__ import (absolute_import, division, print_function)
def is_secret_field(key_name):
    if key_name in secret_fields:
        return True
    return False