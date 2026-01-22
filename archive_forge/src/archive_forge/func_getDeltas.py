from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def getDeltas(self, masterValues, *, round=noRound):
    assert len(masterValues) == len(self.deltaWeights)
    mapping = self.reverseMapping
    out = []
    for i, weights in enumerate(self.deltaWeights):
        delta = masterValues[mapping[i]]
        for j, weight in weights.items():
            if weight == 1:
                delta -= out[j]
            else:
                delta -= out[j] * weight
        out.append(round(delta))
    return out