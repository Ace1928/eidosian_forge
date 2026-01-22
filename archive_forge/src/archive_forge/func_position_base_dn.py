from __future__ import (absolute_import, division, print_function)
import re
def position_base_dn():

    def construct():
        import univention.admin.uldap
        return univention.admin.uldap.position(base_dn())
    return _singleton('position_base_dn', construct)