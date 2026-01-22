from . import storageprotos_pb2 as storageprotos
from .sessionstate import SessionState
def hasSessionState(self, version, aliceBaseKey):
    if self.sessionState.getSessionVersion() == version and aliceBaseKey == self.sessionState.getAliceBaseKey():
        return True
    for state in self.previousStates:
        if state.getSessionVersion() == version and aliceBaseKey == state.getAliceBaseKey():
            return True
    return False