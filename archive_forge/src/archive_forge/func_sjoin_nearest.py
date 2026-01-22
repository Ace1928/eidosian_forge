from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def sjoin_nearest(left_df: GeoDataFrame, right_df: GeoDataFrame, how: str='inner', max_distance: Optional[float]=None, lsuffix: str='left', rsuffix: str='right', distance_col: Optional[str]=None, exclusive: bool=False) -> GeoDataFrame:
    """Spatial join of two GeoDataFrames based on the distance between their geometries.

    Results will include multiple output records for a single input record
    where there are multiple equidistant nearest or intersected neighbors.

    Distance is calculated in CRS units and can be returned using the
    `distance_col` parameter.

    See the User Guide page
    https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
    for more details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    max_distance : float, default None
        Maximum distance within which to query for nearest geometry.
        Must be greater than 0.
        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input
        geometries that are evaluated for nearest items in the tree.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).
    distance_col : string, default None
        If set, save the distances computed between matching geometries under a
        column of this name in the joined GeoDataFrame.
    exclusive : bool, default False
        If True, the nearest geometries that are equal to the input geometry
        will not be returned, default False.
        Requires Shapely >= 2.0.

    Examples
    --------
    >>> import geodatasets
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... )
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... ).to_crs(groceries.crs)

    >>> chicago.head()  # doctest: +SKIP
        ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago)
    >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                    Chain community                              geometry
    0   VIET HOA PLAZA    UPTOWN  MULTIPOINT (1168268.672 1933554.350)
    87      JEWEL OSCO    UPTOWN  MULTIPOINT (1168837.980 1929246.962)


    To include the distances:

    >>> groceries_w_communities = geopandas.sjoin_nearest(groceries, chicago, distance_col="distances")
    >>> groceries_w_communities[["Chain", "community", "distances"]].head(2)  # doctest: +SKIP
                    Chain community  distances
    0   VIET HOA PLAZA    UPTOWN        0.0
    87      JEWEL OSCO    UPTOWN        0.0

    In the following example, we get multiple groceries for Uptown because all
    results are equidistant (in this case zero because they intersect).
    In fact, we get 4 results in total:

    >>> chicago_w_groceries = geopandas.sjoin_nearest(groceries, chicago, distance_col="distances", how="right")
    >>> uptown_results = chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
    >>> uptown_results[["Chain", "community"]]  # doctest: +SKIP
                Chain community
    30  VIET HOA PLAZA    UPTOWN
    30      JEWEL OSCO    UPTOWN
    30          TARGET    UPTOWN
    30       Mariano's    UPTOWN

    See also
    --------
    sjoin : binary predicate joins
    GeoDataFrame.sjoin_nearest : equivalent method

    Notes
    -----
    Since this join relies on distances, results will be inaccurate
    if your geometries are in a geographic CRS.

    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)
    left_df.geometry.values.check_geographic_crs(stacklevel=1)
    right_df.geometry.values.check_geographic_crs(stacklevel=1)
    return_distance = distance_col is not None
    join_df = _nearest_query(left_df, right_df, max_distance, how, return_distance, exclusive)
    if return_distance:
        join_df = join_df.rename(columns={'distances': distance_col})
    else:
        join_df.pop('distances')
    joined = _frame_join(join_df, left_df, right_df, how, lsuffix, rsuffix)
    if return_distance:
        columns = [c for c in joined.columns if c != distance_col] + [distance_col]
        joined = joined[columns]
    return joined