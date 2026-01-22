from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_template_anp_epg(self, epg, fail_module=True):
    """
        Get template endpoint group item that matches the name of an epg.
        :param epg: Name of the epg to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Template epg item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template_anp'])
    kv_list = [KVPair('name', epg)]
    match, existing = self.get_object_from_list(self.schema_objects['template_anp'].details.get('epgs'), kv_list)
    if not match and fail_module:
        msg = "Provided EPG '{0}' not matching existing epg(s): {1}".format(epg, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['template_anp_epg'] = match