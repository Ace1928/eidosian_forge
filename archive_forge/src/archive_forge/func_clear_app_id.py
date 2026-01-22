from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_app_id(self):
    if self.has_app_id_:
        self.has_app_id_ = 0
        self.app_id_ = ''