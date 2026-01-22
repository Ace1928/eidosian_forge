from troveclient.compat import exceptions
def get_public_url(self):
    return '%s/%s' % ('http://localhost:8779/v1.0', self.auth.tenant)