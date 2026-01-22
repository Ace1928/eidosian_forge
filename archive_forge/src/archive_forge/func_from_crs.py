import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
@staticmethod
def from_crs(crs_from: Any, crs_to: Any, always_xy: bool=False, area_of_interest: Optional[AreaOfInterest]=None, authority: Optional[str]=None, accuracy: Optional[float]=None, allow_ballpark: Optional[bool]=None, force_over: bool=False, only_best: Optional[bool]=None) -> 'Transformer':
    """Make a Transformer from a :obj:`pyproj.crs.CRS` or input used to create one.

        See:

        - :c:func:`proj_create_crs_to_crs`
        - :c:func:`proj_create_crs_to_crs_from_pj`

        .. versionadded:: 2.2.0 always_xy
        .. versionadded:: 2.3.0 area_of_interest
        .. versionadded:: 3.1.0 authority, accuracy, allow_ballpark
        .. versionadded:: 3.4.0 force_over
        .. versionadded:: 3.5.0 only_best

        Parameters
        ----------
        crs_from: pyproj.crs.CRS or input used to create one
            Projection of input data.
        crs_to: pyproj.crs.CRS or input used to create one
            Projection of output data.
        always_xy: bool, default=False
            If true, the transform method will accept as input and return as output
            coordinates using the traditional GIS order, that is longitude, latitude
            for geographic CRS and easting, northing for most projected CRS.
        area_of_interest: :class:`.AreaOfInterest`, optional
            The area of interest to help select the transformation.
        authority: str, optional
            When not specified, coordinate operations from any authority will be
            searched, with the restrictions set in the
            authority_to_authority_preference database table related to the
            authority of the source/target CRS themselves. If authority is set
            to “any”, then coordinate operations from any authority will be
            searched. If authority is a non-empty string different from "any",
            then coordinate operations will be searched only in that authority
            namespace (e.g. EPSG).
        accuracy: float, optional
            The minimum desired accuracy (in metres) of the candidate
            coordinate operations.
        allow_ballpark: bool, optional
            Set to False to disallow the use of Ballpark transformation
            in the candidate coordinate operations. Default is to allow.
        force_over: bool, default=False
            If True, it will to force the +over flag on the transformation.
            Requires PROJ 9+.
        only_best: bool, optional
            Can be set to True to cause PROJ to error out if the best
            transformation known to PROJ and usable by PROJ if all grids known and
            usable by PROJ were accessible, cannot be used. Best transformation should
            be understood as the transformation returned by
            :c:func:`proj_get_suggested_operation` if all known grids were
            accessible (either locally or through network).
            Note that the default value for this option can be also set with the
            :envvar:`PROJ_ONLY_BEST_DEFAULT` environment variable, or with the
            ``only_best_default`` setting of :ref:`proj-ini`.
            The only_best kwarg overrides the default value if set.
            Requires PROJ 9.2+.

        Returns
        -------
        Transformer

        """
    return Transformer(TransformerFromCRS(cstrencode(CRS.from_user_input(crs_from).srs), cstrencode(CRS.from_user_input(crs_to).srs), always_xy=always_xy, area_of_interest=area_of_interest, authority=authority, accuracy=accuracy if accuracy is None else str(accuracy), allow_ballpark=allow_ballpark, force_over=force_over, only_best=only_best))