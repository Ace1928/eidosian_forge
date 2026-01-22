import random
from ._fold_storage import FoldStorage
from ._fold_storage import _FoldFile
def create_fold(self, fold_set, name, id):
    file_name = self.create_name_from_id(name, id)
    fold_file = _FoldFile(fold_set, file_name, sep=self._line_reader.get_separator(), column_description=self._column_description)
    self._folds_storage.add(fold_file)
    return fold_file