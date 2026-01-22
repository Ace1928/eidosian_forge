from __future__ import (absolute_import, division, print_function)
import re
def umc_module_for_edit(module, object_dn, superordinate=None):
    """Returns an UMC module object prepared for editing an existing entry.

   The module is a module specification according to the udm commandline.
   Example values are:
       * users/user
       * shares/share
       * groups/group

   The object_dn MUST be the dn of the object itself, not the container!
   """
    mod = module_by_name(module)
    objects = get_umc_admin_objects()
    position = position_base_dn()
    position.setDn(ldap_dn_tree_parent(object_dn))
    obj = objects.get(mod, config(), uldap(), position=position, superordinate=superordinate, dn=object_dn)
    obj.open()
    return obj