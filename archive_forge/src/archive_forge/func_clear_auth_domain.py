from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_auth_domain(self):
    if self.has_auth_domain_:
        self.has_auth_domain_ = 0
        self.auth_domain_ = ''