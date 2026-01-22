from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_site_anp_epg(self, epg_name, fail_module=True):
    """
        Get site anp epg item that matches the epgs.
        :param epg: epg to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Site anp epg item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['site_anp', 'template_anp_epg'])
    kv_list = [KVPair('epgRef', self.schema_objects['template_anp_epg'].details.get('epgRef'))]
    match, existing = self.get_object_from_list(self.schema_objects['site_anp'].details.get('epgs'), kv_list)
    if not match and fail_module:
        msg = "Provided EPG '{0}' not matching existing site anp epg(s): {1}".format(epg_name, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['site_anp_epg'] = match