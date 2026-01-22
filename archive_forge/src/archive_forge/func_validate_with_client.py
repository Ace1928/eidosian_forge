from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
def validate_with_client(self, client, value):
    params = {}
    neutron_client = client.client(CLIENT_NAME)
    if self.service_type:
        params['service_type'] = self.service_type
    providers = neutron_client.list_service_providers(retrieve_all=True, **params)['service_providers']
    names = [provider['name'] for provider in providers]
    if value not in names:
        not_found_message = _("Unable to find neutron provider '%(provider)s', available providers are %(providers)s.") % {'provider': value, 'providers': names}
        raise exception.StackValidationFailed(message=not_found_message)