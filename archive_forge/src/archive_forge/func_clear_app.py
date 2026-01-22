from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_app(self):
    if self.has_app_:
        self.has_app_ = 0
        self.app_ = ''