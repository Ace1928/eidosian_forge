from typing import Hashable, MutableMapping, Tuple
import numpy as np
import stanio
def build_xarray_data(data: MutableMapping[Hashable, Tuple[Tuple[str, ...], np.ndarray]], var: stanio.Variable, drawset: np.ndarray) -> None:
    """
    Adds Stan variable name, labels, and values to a dictionary
    that will be used to construct an xarray DataSet.
    """
    var_dims: Tuple[str, ...] = ('draw', 'chain')
    var_dims += tuple((f'{var.name}_dim_{i}' for i in range(len(var.dimensions))))
    data[var.name] = (var_dims, var.extract_reshape(drawset))