import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def createVariable(self, name, type, dimensions):
    """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
    shape = tuple([self.dimensions[dim] for dim in dimensions])
    shape_ = tuple([dim or 0 for dim in shape])
    type = dtype(type)
    typecode, size = (type.char, type.itemsize)
    if (typecode, size) not in REVERSE:
        raise ValueError('NetCDF 3 does not support type %s' % type)
    data = empty(shape_, dtype=type.newbyteorder('B'))
    self.variables[name] = netcdf_variable(data, typecode, size, shape, dimensions, maskandscale=self.maskandscale)
    return self.variables[name]