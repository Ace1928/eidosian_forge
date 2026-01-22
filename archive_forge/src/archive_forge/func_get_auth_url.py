from keystoneauth1 import exceptions
def get_auth_url(self, sp_id):
    return self._get_service_provider(sp_id).get('auth_url')