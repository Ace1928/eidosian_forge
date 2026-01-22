from __future__ import (absolute_import, division, print_function)
import re
def module_by_name(module_name_):
    """Returns an initialized UMC module, identified by the given name.

    The module is a module specification according to the udm commandline.
    Example values are:
    * users/user
    * shares/share
    * groups/group

    If the module does not exist, a KeyError is raised.

    The modules are cached, so they won't be re-initialized
    in subsequent calls.
    """

    def construct():
        import univention.admin.modules
        init_modules()
        module = univention.admin.modules.get(module_name_)
        univention.admin.modules.init(uldap(), position_base_dn(), module)
        return module
    return _singleton('module/%s' % module_name_, construct)