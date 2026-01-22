from yowsup.structs import  ProtocolTreeNode
from .iq import IqProtocolEntity
def setErrorProps(self, code, text, backoff):
    self.code = code
    self.text = text
    self.backoff = int(backoff) if backoff else 0