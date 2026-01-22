def carshare():
    """
    Each row represents the availability of car-sharing services near the centroid of a zone
    in Montreal over a month-long period.

    Returns:
        A `pandas.DataFrame` with 249 rows and the following columns:
        `['centroid_lat', 'centroid_lon', 'car_hours', 'peak_hour']`."""
    return _get_dataset('carshare')