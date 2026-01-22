from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_meaning_uri(self):
    if self.has_meaning_uri_:
        self.has_meaning_uri_ = 0
        self.meaning_uri_ = ''