from datetime import datetime
import sys
@classmethod
def login_with(cls, consumer_name, service_root=None, launchpadlib_dir=None, timeout=None, proxy_info=None):
    """Log in to Launchpad with possibly cached credentials."""
    from launchpadlib.testing.resources import get_application
    return cls(object(), application=get_application())