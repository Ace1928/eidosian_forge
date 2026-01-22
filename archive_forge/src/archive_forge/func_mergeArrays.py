from io import StringIO
from antlr4.error.Errors import IllegalStateException
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNState import ATNState
def mergeArrays(a: ArrayPredictionContext, b: ArrayPredictionContext, rootIsWildcard: bool, mergeCache: dict):
    if mergeCache is not None:
        previous = mergeCache.get((a, b), None)
        if previous is not None:
            return previous
        previous = mergeCache.get((b, a), None)
        if previous is not None:
            return previous
    i = 0
    j = 0
    k = 0
    mergedReturnStates = [None] * (len(a.returnStates) + len(b.returnStates))
    mergedParents = [None] * len(mergedReturnStates)
    while i < len(a.returnStates) and j < len(b.returnStates):
        a_parent = a.parents[i]
        b_parent = b.parents[j]
        if a.returnStates[i] == b.returnStates[j]:
            payload = a.returnStates[i]
            bothDollars = payload == PredictionContext.EMPTY_RETURN_STATE and a_parent is None and (b_parent is None)
            ax_ax = (a_parent is not None and b_parent is not None) and a_parent == b_parent
            if bothDollars or ax_ax:
                mergedParents[k] = a_parent
                mergedReturnStates[k] = payload
            else:
                mergedParent = merge(a_parent, b_parent, rootIsWildcard, mergeCache)
                mergedParents[k] = mergedParent
                mergedReturnStates[k] = payload
            i += 1
            j += 1
        elif a.returnStates[i] < b.returnStates[j]:
            mergedParents[k] = a_parent
            mergedReturnStates[k] = a.returnStates[i]
            i += 1
        else:
            mergedParents[k] = b_parent
            mergedReturnStates[k] = b.returnStates[j]
            j += 1
        k += 1
    if i < len(a.returnStates):
        for p in range(i, len(a.returnStates)):
            mergedParents[k] = a.parents[p]
            mergedReturnStates[k] = a.returnStates[p]
            k += 1
    else:
        for p in range(j, len(b.returnStates)):
            mergedParents[k] = b.parents[p]
            mergedReturnStates[k] = b.returnStates[p]
            k += 1
    if k < len(mergedParents):
        if k == 1:
            merged = SingletonPredictionContext.create(mergedParents[0], mergedReturnStates[0])
            if mergeCache is not None:
                mergeCache[a, b] = merged
            return merged
        mergedParents = mergedParents[0:k]
        mergedReturnStates = mergedReturnStates[0:k]
    merged = ArrayPredictionContext(mergedParents, mergedReturnStates)
    if merged == a:
        if mergeCache is not None:
            mergeCache[a, b] = a
        return a
    if merged == b:
        if mergeCache is not None:
            mergeCache[a, b] = b
        return b
    combineCommonParents(mergedParents)
    if mergeCache is not None:
        mergeCache[a, b] = merged
    return merged