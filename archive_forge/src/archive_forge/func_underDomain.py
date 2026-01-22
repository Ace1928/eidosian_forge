import re
def underDomain(self, domain):
    """Return True if the given domain name a parent of the URL's host."""
    if len(domain) == 0:
        return True
    our_segments = self.host.split('.')
    domain_segments = domain.split('.')
    return our_segments[-len(domain_segments):] == domain_segments