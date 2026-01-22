from typing import Any, Callable, Dict, Hashable, Optional, Sequence, Tuple, Union, cast
from ..config import registry
from ..model import Model
from ..types import DTypes, Ints1d, Ints2d
from ..util import is_xp_array, to_numpy
@registry.layers('remap_ids.v1')
def remap_ids(mapping_table: Dict[Any, int]={}, default: int=0, dtype: DTypes='i') -> Model[InT_v1, OutT_v1]:
    """Remap string or integer inputs using a mapping table, usually as a
    preprocess before embeddings. The mapping table can be passed in on input,
    or updated after the layer has been created. The mapping table is stored in
    the "mapping_table" attribute.
    """
    return Model('remap_ids', forward, attrs={'mapping_table': mapping_table, 'dtype': dtype, 'default': default})