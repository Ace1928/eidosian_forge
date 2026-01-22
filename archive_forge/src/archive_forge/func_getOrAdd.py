from io import StringIO
from functools import reduce
from antlr4.PredictionContext import PredictionContext, merge
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.SemanticContext import SemanticContext
from antlr4.error.Errors import UnsupportedOperationException, IllegalStateException
def getOrAdd(self, config: ATNConfig):
    h = config.hashCodeForConfigSet()
    l = self.configLookup.get(h, None)
    if l is not None:
        r = next((cfg for cfg in l if config.equalsForConfigSet(cfg)), None)
        if r is not None:
            return r
    if l is None:
        l = [config]
        self.configLookup[h] = l
    else:
        l.append(config)
    return config