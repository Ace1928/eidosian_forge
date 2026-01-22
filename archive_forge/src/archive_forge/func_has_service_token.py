from keystoneauth1.identity import base as base_identity
@property
def has_service_token(self):
    """Did this authentication request contained a service token."""
    return self.service is not None