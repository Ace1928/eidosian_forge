import numpy
from rdkit.ML.Data import Quantize
def trainModel(self):
    """ We will assume at this point that the training examples have been set

        We have to estmate the conditional probabilities for each of the (binned) descriptor
        component give a outcome (or class). Also the probabilities for each class is estimated
        """
    n = len(self._trainingExamples)
    for i in range(self._nClasses):
        self._classProbs[i] = 0.0
    if not self._useSigs and max(self._qBounds) > 0:
        self._computeQuantBounds()
    ncls = {}
    incr = 1.0 / n
    for eg in self._trainingExamples:
        cls = eg[-1]
        self._classProbs[cls] += incr
        ncls[cls] = ncls.get(cls, 0) + 1
        tmp = self._condProbs[cls]
        if not self._useSigs:
            for ai in self._attrs:
                bid = eg[ai]
                if self._qBounds[ai] > 0:
                    bid = _getBinId(bid, self._QBoundVals[ai])
                tmp[ai][bid] += 1.0
        else:
            for ai in self._attrs:
                if eg[1].GetBit(ai):
                    tmp[ai][1] += 1.0
                else:
                    tmp[ai][0] += 1.0
    for cls in range(self._nClasses):
        if cls not in ncls:
            continue
        tmp = self._condProbs[cls]
        for ai in self._attrs:
            if not self._useSigs:
                nbnds = self._nPosVals[ai]
                if self._qBounds[ai] > 0:
                    nbnds = self._qBounds[ai]
            else:
                nbnds = 2
            for bid in range(nbnds):
                if self._mEstimateVal <= 0.0:
                    tmp[ai][bid] /= ncls[cls]
                else:
                    pdesc = 0.0
                    if self._qBounds[ai] > 0:
                        pdesc = 1.0 / (1 + len(self._QBoundVals[ai]))
                    elif self._nPosVals[ai] > 0:
                        pdesc = 1.0 / self._nPosVals[ai]
                    else:
                        raise ValueError('Neither Bounds set nor data pre-quantized for attribute ' + str(ai))
                    tmp[ai][bid] += self._mEstimateVal * pdesc
                    tmp[ai][bid] /= ncls[cls] + self._mEstimateVal