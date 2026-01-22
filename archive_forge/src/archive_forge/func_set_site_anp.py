from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_site_anp(self, anp_name, fail_module=True):
    """
        Get site application profile item that matches the name of a anp.
        :param anp_name: Name of the anp to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Site anp item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template_anp', 'site'])
    kv_list = [KVPair('anpRef', self.schema_objects['template_anp'].details.get('anpRef'))]
    match, existing = self.get_object_from_list(self.schema_objects['site'].details.get('anps'), kv_list)
    if not match and fail_module:
        msg = "Provided ANP '{0}' not matching existing site anp(s): {1}".format(anp_name, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['site_anp'] = match