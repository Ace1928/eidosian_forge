import time
@property
def mean_units_processed(self):
    if len(self._units_processed) == 0:
        return 0.0
    return float(sum(self._units_processed)) / len(self._units_processed)