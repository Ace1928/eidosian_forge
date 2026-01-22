from __future__ import (absolute_import, division, print_function)
import re
def config_registry():

    def construct():
        import univention.config_registry
        ucr = univention.config_registry.ConfigRegistry()
        ucr.load()
        return ucr
    return _singleton('config_registry', construct)