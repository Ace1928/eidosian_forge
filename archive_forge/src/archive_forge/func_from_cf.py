import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
@staticmethod
def from_cf(in_cf: dict, ellipsoidal_cs: Optional[Any]=None, cartesian_cs: Optional[Any]=None, vertical_cs: Optional[Any]=None) -> 'CRS':
    """
        .. versionadded:: 2.2.0

        .. versionadded:: 3.0.0 ellipsoidal_cs, cartesian_cs, vertical_cs

        This converts a Climate and Forecast (CF) Grid Mapping Version 1.8
        dict to a :obj:`pyproj.crs.CRS` object.

        :ref:`build_crs_cf`

        Parameters
        ----------
        in_cf: dict
            CF version of the projection.
        ellipsoidal_cs: Any, optional
            Input to create an Ellipsoidal Coordinate System.
            Anything accepted by :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or an Ellipsoidal Coordinate System created from :ref:`coordinate_system`.
        cartesian_cs: Any, optional
            Input to create a Cartesian Coordinate System.
            Anything accepted by :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or :class:`pyproj.crs.coordinate_system.Cartesian2DCS`.
        vertical_cs: Any, optional
            Input to create a Vertical Coordinate System accepted by
            :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or :class:`pyproj.crs.coordinate_system.VerticalCS`

        Returns
        -------
        CRS
        """
    unknown_names = ('unknown', 'undefined')
    if 'crs_wkt' in in_cf:
        return CRS(in_cf['crs_wkt'])
    if 'spatial_ref' in in_cf:
        return CRS(in_cf['spatial_ref'])
    grid_mapping_name = in_cf.get('grid_mapping_name')
    if grid_mapping_name is None:
        raise CRSError("CF projection parameters missing 'grid_mapping_name'")
    datum = _horizontal_datum_from_params(in_cf)
    try:
        geographic_conversion_method: Optional[Callable] = _GEOGRAPHIC_GRID_MAPPING_NAME_MAP[grid_mapping_name]
    except KeyError:
        geographic_conversion_method = None
    geographic_crs_name = in_cf.get('geographic_crs_name')
    if datum:
        geographic_crs: CRS = GeographicCRS(name=geographic_crs_name or 'undefined', datum=datum, ellipsoidal_cs=ellipsoidal_cs)
    elif geographic_crs_name and geographic_crs_name not in unknown_names:
        geographic_crs = CRS(geographic_crs_name)
        if ellipsoidal_cs is not None:
            geographic_crs_json = geographic_crs.to_json_dict()
            geographic_crs_json['coordinate_system'] = CoordinateSystem.from_user_input(ellipsoidal_cs).to_json_dict()
            geographic_crs = CRS(geographic_crs_json)
    else:
        geographic_crs = GeographicCRS(ellipsoidal_cs=ellipsoidal_cs)
    if grid_mapping_name == 'latitude_longitude':
        return geographic_crs
    if geographic_conversion_method is not None:
        return DerivedGeographicCRS(base_crs=geographic_crs, conversion=geographic_conversion_method(in_cf), ellipsoidal_cs=ellipsoidal_cs)
    try:
        conversion_method = _GRID_MAPPING_NAME_MAP[grid_mapping_name]
    except KeyError:
        raise CRSError(f'Unsupported grid mapping name: {grid_mapping_name}') from None
    projected_crs = ProjectedCRS(name=in_cf.get('projected_crs_name', 'undefined'), conversion=conversion_method(in_cf), geodetic_crs=geographic_crs, cartesian_cs=cartesian_cs)
    bound_crs = None
    if 'towgs84' in in_cf:
        bound_crs = BoundCRS(source_crs=projected_crs, target_crs='WGS 84', transformation=ToWGS84Transformation(projected_crs.geodetic_crs, *_try_list_if_string(in_cf['towgs84'])))
    if 'geopotential_datum_name' not in in_cf:
        return bound_crs or projected_crs
    vertical_crs = VerticalCRS(name='undefined', datum=in_cf['geopotential_datum_name'], geoid_model=in_cf.get('geoid_name'), vertical_cs=vertical_cs)
    return CompoundCRS(name='undefined', components=[bound_crs or projected_crs, vertical_crs])