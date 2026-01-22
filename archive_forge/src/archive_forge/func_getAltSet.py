from io import StringIO
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.SemanticContext import SemanticContext
def getAltSet(self):
    if self.configs is not None:
        return set((cfg.alt for cfg in self.configs)) or None
    return None