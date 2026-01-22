import re
import warnings
from typing import Any, Optional
from pyproj._compat import cstrencode
from pyproj._transformer import Factors
from pyproj.crs import CRS
from pyproj.enums import TransformDirection
from pyproj.list import get_proj_operations_map
from pyproj.transformer import Transformer, TransformerFromPipeline
from pyproj.utils import _convertback, _copytobuffer
def definition_string(self) -> str:
    """Returns formal definition string for projection

        >>> Proj("EPSG:4326").definition_string()
        'proj=longlat datum=WGS84 no_defs ellps=WGS84 towgs84=0,0,0'
        """
    return self.definition