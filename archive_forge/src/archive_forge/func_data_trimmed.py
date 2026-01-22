import numbers
import numpy as np
@property
def data_trimmed(self):
    """numpy array of trimmed and sorted data
        """
    return self.data_sorted[self.sl]