from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_site_bd(self, bd_name, fail_module=True):
    """
        Get site bridge domain item that matches the name of a bd.
        :param bd_name: Name of the bd to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Site bd item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template', 'site'])
    kv_list = [KVPair('bdRef', self.mso.bd_ref(schema_id=self.id, template=self.schema_objects['template'].details.get('name'), bd=bd_name))]
    match, existing = self.get_object_from_list(self.schema_objects['site'].details.get('bds'), kv_list)
    if not match and fail_module:
        msg = "Provided BD '{0}' not matching existing site bd(s): {1}".format(bd_name, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['site_bd'] = match