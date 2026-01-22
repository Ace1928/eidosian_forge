from typing import TYPE_CHECKING
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
def fit_status(self):
    fittable_count = 0
    fitted_count = 0
    for p in self.preprocessors:
        if p.fit_status() == Preprocessor.FitStatus.FITTED:
            fittable_count += 1
            fitted_count += 1
        elif p.fit_status() in (Preprocessor.FitStatus.NOT_FITTED, Preprocessor.FitStatus.PARTIALLY_FITTED):
            fittable_count += 1
        else:
            assert p.fit_status() == Preprocessor.FitStatus.NOT_FITTABLE
    if fittable_count > 0:
        if fitted_count == fittable_count:
            return Preprocessor.FitStatus.FITTED
        elif fitted_count > 0:
            return Preprocessor.FitStatus.PARTIALLY_FITTED
        else:
            return Preprocessor.FitStatus.NOT_FITTED
    else:
        return Preprocessor.FitStatus.NOT_FITTABLE