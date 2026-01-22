from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def mergeSingletons(a: SingletonPredictionContext, b: SingletonPredictionContext, rootIsWildcard: bool, mergeCache: dict):
    if mergeCache is not None:
        previous = mergeCache.get((a, b), None)
        if previous is not None:
            return previous
        previous = mergeCache.get((b, a), None)
        if previous is not None:
            return previous
    merged = mergeRoot(a, b, rootIsWildcard)
    if merged is not None:
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged
    if a.returnState == b.returnState:
        parent = merge(a.parentCtx, b.parentCtx, rootIsWildcard, mergeCache)
        if parent == a.parentCtx:
            return a
        if parent == b.parentCtx:
            return b
        merged = SingletonPredictionContext.create(parent, a.returnState)
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged
    else:
        singleParent = None
        if a is b or (a.parentCtx is not None and a.parentCtx == b.parentCtx):
            singleParent = a.parentCtx
        if singleParent is not None:
            payloads = [a.returnState, b.returnState]
            if a.returnState > b.returnState:
                payloads = [b.returnState, a.returnState]
            parents = [singleParent, singleParent]
            merged = ArrayPredictionContext(parents, payloads)
            if mergeCache is not None:
                mergeCache[a, b] = merged
            return merged
        payloads = [a.returnState, b.returnState]
        parents = [a.parentCtx, b.parentCtx]
        if a.returnState > b.returnState:
            payloads = [b.returnState, a.returnState]
            parents = [b.parentCtx, a.parentCtx]
        merged = ArrayPredictionContext(parents, payloads)
        if mergeCache is not None:
            mergeCache[a, b] = merged
        return merged