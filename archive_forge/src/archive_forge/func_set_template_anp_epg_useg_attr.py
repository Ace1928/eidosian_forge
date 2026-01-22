from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_template_anp_epg_useg_attr(self, useg_attr, fail_module=True):
    """
        Get template endpoint group item that matches the name of an EPG uSeg Attribute.
        :param useg_attr: Name of the EPG uSeg Attribute to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Template EPG uSeg Attribute item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template_anp_epg'])
    kv_list = [KVPair('name', useg_attr)]
    match, existing = self.get_object_from_list(self.schema_objects['template_anp_epg'].details.get('uSegAttrs'), kv_list)
    if not match and fail_module:
        msg = "Provided uSeg Attribute '{0}' does not match the existing uSeg Attribute(s): {1}".format(useg_attr, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['template_anp_epg_useg_attribute'] = match