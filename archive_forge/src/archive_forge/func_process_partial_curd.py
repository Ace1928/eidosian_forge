from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def process_partial_curd(self, argument_specs=None):
    self.metadata = argument_specs
    module_name = self.module_level2_name
    params = self.module.params
    track = [module_name]
    if not params.get('bypass_validation', False):
        self.check_versioning_mismatch(track, argument_specs.get(module_name, None), params.get(module_name, None))
    adom_value = params.get('adom', None)
    target_url = self._get_target_url(adom_value, self.jrpc_urls)
    target_url = self._get_replaced_url(target_url)
    target_url = target_url.rstrip('/')
    api_params = {'url': target_url}
    if module_name in params:
        params = remove_aliases(params, argument_specs)
        api_params[self.top_level_schema_name] = params[module_name]
    response = self.conn.send_request(self._propose_method('set'), [api_params])
    self.do_exit(response)