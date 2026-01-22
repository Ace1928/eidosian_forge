from ..io import DataIter, DataDesc
from .. import ndarray as nd
def getlabel(self):
    if self.getpad():
        lshape = self._current_batch[1].shape
        ret = nd.empty(shape=[self.batch_size] + list(lshape[1:]))
        ret[:lshape[0]] = self._current_batch[1].astype(self.dtype)
        return [ret]
    return [self._current_batch[1].astype(self.dtype)]