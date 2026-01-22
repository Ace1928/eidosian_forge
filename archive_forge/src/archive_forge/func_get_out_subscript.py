import warnings
import numpy as np
import xarray as xr
def get_out_subscript(self):
    if not self.out_subscript:
        return ''
    return f'->{self.out_subscript}'