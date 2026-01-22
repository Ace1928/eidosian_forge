from enum import Enum
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNState import RuleStopState
from antlr4.atn.SemanticContext import SemanticContext
@classmethod
def hasSLLConflictTerminatingPrediction(cls, mode: PredictionMode, configs: ATNConfigSet):
    if cls.allConfigsInRuleStopStates(configs):
        return True
    if mode == PredictionMode.SLL:
        if configs.hasSemanticContext:
            dup = ATNConfigSet()
            for c in configs:
                c = ATNConfig(config=c, semantic=SemanticContext.NONE)
                dup.add(c)
            configs = dup
    altsets = cls.getConflictingAltSubsets(configs)
    return cls.hasConflictingAltSet(altsets) and (not cls.hasStateAssociatedWithOneAlt(configs))