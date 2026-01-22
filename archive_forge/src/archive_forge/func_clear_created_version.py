from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_created_version(self):
    if self.has_created_version_:
        self.has_created_version_ = 0
        self.created_version_ = 0