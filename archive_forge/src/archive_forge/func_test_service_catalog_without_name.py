import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_without_name(self):
    f = fixture.V3Token(audit_chain_id=uuid.uuid4().hex)
    if not f.project_id:
        f.set_project_scope()
    f.add_role(name='admin')
    f.add_role(name='member')
    region = 'RegionOne'
    tenant = '225da22d3ce34b15877ea70b2a575f58'
    s = f.add_service('volume')
    s.add_standard_endpoints(public='http://public.com:8776/v1/%s' % tenant, internal='http://internal:8776/v1/%s' % tenant, admin='http://admin:8776/v1/%s' % tenant, region=region)
    s = f.add_service('image')
    s.add_standard_endpoints(public='http://public.com:9292/v1', internal='http://internal:9292/v1', admin='http://admin:9292/v1', region=region)
    s = f.add_service('compute')
    s.add_standard_endpoints(public='http://public.com:8774/v2/%s' % tenant, internal='http://internal:8774/v2/%s' % tenant, admin='http://admin:8774/v2/%s' % tenant, region=region)
    s = f.add_service('ec2')
    s.add_standard_endpoints(public='http://public.com:8773/services/Cloud', internal='http://internal:8773/services/Cloud', admin='http://admin:8773/services/Admin', region=region)
    s = f.add_service('identity')
    s.add_standard_endpoints(public='http://public.com:5000/v3', internal='http://internal:5000/v3', admin='http://admin:35357/v3', region=region)
    pr_auth_ref = access.create(body=f)
    pr_sc = pr_auth_ref.service_catalog
    url_ref = 'http://public.com:8774/v2/225da22d3ce34b15877ea70b2a575f58'
    url = pr_sc.url_for(service_type='compute', service_name='NotExist', interface='public')
    self.assertEqual(url_ref, url)
    ab_auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    ab_sc = ab_auth_ref.service_catalog
    self.assertRaises(exceptions.EndpointNotFound, ab_sc.url_for, service_type='compute', service_name='NotExist', interface='public')