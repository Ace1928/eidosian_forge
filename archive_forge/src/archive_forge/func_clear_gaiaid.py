from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_gaiaid(self):
    if self.has_gaiaid_:
        self.has_gaiaid_ = 0
        self.gaiaid_ = 0