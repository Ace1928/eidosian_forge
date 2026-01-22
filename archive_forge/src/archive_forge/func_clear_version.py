from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_version(self):
    if self.has_version_:
        self.has_version_ = 0
        self.version_ = 0