from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_meaning(self):
    if self.has_meaning_:
        self.has_meaning_ = 0
        self.meaning_ = 0