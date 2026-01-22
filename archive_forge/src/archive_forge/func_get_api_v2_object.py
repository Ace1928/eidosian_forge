from __future__ import absolute_import, division, print_function
from .controller_api import ControllerModule
from ansible.module_utils.basic import missing_required_lib
def get_api_v2_object(self):
    if not self.apiV2Ref:
        if not self.authenticated:
            self.authenticate()
        v2_index = get_registered_page('/api/v2/')(self.connection).get()
        self.api_ref = ApiV2(connection=self.connection, **{'json': v2_index})
    return self.api_ref