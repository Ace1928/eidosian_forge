from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_x(self):
    if self.has_x_:
        self.has_x_ = 0
        self.x_ = 0.0