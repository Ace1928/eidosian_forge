import uuid
import fixtures
from keystoneauth1.fixture import v2
from keystoneauth1.fixture import v3
import os_service_types
def build_tokens(self):
    self.clear_tokens()
    for service in _service_type_manager.services:
        service_type = service['service_type']
        if service_type == 'ec2-api':
            continue
        service_name = service['project']
        ets = self._get_endpoint_templates(service_type)
        v3_svc = self.v3_token.add_service(service_type, name=service_name)
        v2_svc = self.v2_token.add_service(service_type, name=service_name)
        v3_svc.add_standard_endpoints(region='RegionOne', **ets)
        if service_type == 'identity':
            ets = self._get_endpoint_templates(service_type, v2=True)
        v2_svc.add_endpoint(region='RegionOne', **ets)
        for alias in service.get('aliases', []):
            ets = self._get_endpoint_templates(service_type, alias=alias)
            v3_svc = self.v3_token.add_service(alias, name=service_name)
            v2_svc = self.v2_token.add_service(alias, name=service_name)
            v3_svc.add_standard_endpoints(region='RegionOne', **ets)
            v2_svc.add_endpoint(region='RegionOne', **ets)