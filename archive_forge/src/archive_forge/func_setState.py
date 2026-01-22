from . import storageprotos_pb2 as storageprotos
from .sessionstate import SessionState
def setState(self, sessionState):
    self.sessionState = sessionState