from datetime import datetime
import sys
@classmethod
def get_token_and_login(cls, consumer_name, service_root=None, cache=None, timeout=None, proxy_info=None):
    """Get credentials from Launchpad and log into the service root."""
    from launchpadlib.testing.resources import get_application
    return cls(object(), application=get_application())