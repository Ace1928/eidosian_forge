from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_nickname(self):
    if self.has_nickname_:
        self.has_nickname_ = 0
        self.nickname_ = ''