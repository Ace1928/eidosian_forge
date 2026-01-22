from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_obfuscated_gaiaid(self):
    if self.has_obfuscated_gaiaid_:
        self.has_obfuscated_gaiaid_ = 0
        self.obfuscated_gaiaid_ = ''