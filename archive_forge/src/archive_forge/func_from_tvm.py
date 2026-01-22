from collections import OrderedDict
import numpy as _np
@classmethod
def from_tvm(cls, x):
    """Build a ConfigSpace from autotvm.ConfigSpace

        Parameters
        ----------
        cls: class
            Calling class
        x: autotvm.ConfigSpace
            The source object

        Returns
        -------
        ret: ConfigSpace
            The corresponding ConfigSpace object
        """
    space_map = OrderedDict([(k, OtherOptionSpace.from_tvm(v)) for k, v in x.space_map.items()])
    _entity_map = OrderedDict([(k, OtherOptionEntity.from_tvm(v)) for k, v in x._entity_map.items()])
    return cls(space_map, _entity_map)