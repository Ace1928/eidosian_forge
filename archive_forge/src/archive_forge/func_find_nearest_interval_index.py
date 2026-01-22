import bisect
def find_nearest_interval_index(interval_array, target, tolerance=None, prefer_left=True):
    array_lo = 0
    array_hi = len(interval_array)
    target_tuple = (target,)
    i = bisect.bisect_right(interval_array, target_tuple, lo=array_lo, hi=array_hi)
    distance_tol = 0.0 if tolerance is None else tolerance
    if i == array_lo:
        nearest_index = i
        delta = _distance_from_interval(target, interval_array[i], tolerance=distance_tol)
    elif i == array_hi:
        nearest_index = i - 1
        delta = _distance_from_interval(target, interval_array[i - 1], tolerance=distance_tol)
    elif prefer_left:
        delta, nearest_index = min(((_distance_from_interval(target, interval_array[j], tolerance=distance_tol), j) for j in [i - 1, i]))
    else:
        delta, neg_nearest_index = min(((_distance_from_interval(target, interval_array[j], tolerance=distance_tol), -j) for j in [i - 1, i]))
        nearest_index = -neg_nearest_index
    if prefer_left and nearest_index >= array_lo + 1:
        delta_left = _distance_from_interval(target, interval_array[nearest_index - 1], tolerance=distance_tol)
        if delta_left <= delta:
            nearest_index = nearest_index - 1
            delta = delta_left
    elif not prefer_left and nearest_index < array_hi - 1:
        delta_right = _distance_from_interval(target, interval_array[nearest_index + 1], tolerance=distance_tol)
        if delta_right <= delta:
            nearest_index = nearest_index + 1
            delta = delta_right
    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index