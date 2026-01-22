import numpy as np
from .. import h5s
def read_dtypes(dataset_dtype, names):
    """ Returns a 2-tuple containing:

    1. Output dataset dtype
    2. Dtype containing HDF5-appropriate description of destination
    """
    if len(names) == 0:
        format_dtype = dataset_dtype
    elif dataset_dtype.names is None:
        raise ValueError('Field names only allowed for compound types')
    elif any((x not in dataset_dtype.names for x in names)):
        raise ValueError('Field does not appear in this type.')
    else:
        format_dtype = np.dtype([(name, dataset_dtype.fields[name][0]) for name in names])
    if len(names) == 1:
        output_dtype = format_dtype.fields[names[0]][0]
    else:
        output_dtype = format_dtype
    return (output_dtype, format_dtype)