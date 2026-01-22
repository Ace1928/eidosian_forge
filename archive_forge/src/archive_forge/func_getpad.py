from ..io import DataIter, DataDesc
from .. import ndarray as nd
def getpad(self):
    return self.batch_size - self._current_batch[0].shape[0]