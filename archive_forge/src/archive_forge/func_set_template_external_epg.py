from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_template_external_epg(self, external_epg, fail_module=True):
    """
        Get template external epg item that matches the name of an anp.
        :param anp: Name of the anp to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Template anp item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template'])
    kv_list = [KVPair('name', external_epg)]
    match, existing = self.get_object_from_list(self.schema_objects['template'].details.get('externalEpgs'), kv_list)
    if not match and fail_module:
        msg = "Provided External EPG '{0}' not matching existing external_epg(s): {1}".format(external_epg, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['template_external_epg'] = match