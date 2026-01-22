import logging
from osc_lib.command import command
from osc_lib import utils
from monascaclient import version
@property
def mon_url(self):
    if self._endpoint:
        return self._endpoint
    app_args = self.app_args
    cm = self.app.client_manager
    endpoint = app_args.monasca_api_url
    if not endpoint:
        req_data = {'service_type': 'monitoring', 'region_name': cm.region_name, 'interface': cm.interface}
        LOG.debug('Discovering monasca endpoint using %s' % req_data)
        endpoint = cm.get_endpoint_for_service_type(**req_data)
    else:
        LOG.debug('Using supplied endpoint=%s' % endpoint)
    self._endpoint = endpoint
    return self._endpoint