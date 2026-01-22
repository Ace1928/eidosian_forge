from __future__ import (absolute_import, division, print_function)
import re
def get_umc_admin_objects():
    """Convenience accessor for getting univention.admin.objects.

    This implements delayed importing, so the univention.* modules
    are not loaded until this function is called.
    """
    import univention.admin
    return univention.admin.objects