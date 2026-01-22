from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def segments_meet_not_at_endpoint(ab, uv):
    """
    Does the interior of either segment meet any point of the other
    one?

    >>> p = Vector3([1, 1, 0])
    >>> q = Vector3([3, -1,0])
    >>> a = Vector3([2, 0, 0])
    >>> b = Vector3([0, 2, 0])
    >>> c = Vector3([1, 0, 0])
    >>> o = Vector3([0, 0, 0])
    >>> segments_meet_not_at_endpoint((p, 2*p), (2*p, 3*p))
    False
    >>> segments_meet_not_at_endpoint((p, 3*p), (2*p, 4*p))
    True
    >>> segments_meet_not_at_endpoint((p, 2*p), (3*p, q))
    False
    >>> segments_meet_not_at_endpoint((p, 3*p), (2*p, q))
    True
    >>> segments_meet_not_at_endpoint((-a, a), (b + c, -b + c))
    True
    >>> segments_meet_not_at_endpoint((-a, a), (a, -b + c))
    False
    >>> segments_meet_not_at_endpoint((o, a), (-p, p))
    True
    >>> segments_meet_not_at_endpoint((-p + q, p + q), (-a + q, a + q))
    True

    >>> b, c = Vector3([0, 0, 1]), Vector3([0, 1, 0])
    >>> u, v = Vector3([-1, 0, 1]), Vector3([-1, 1, 2])/3
    >>> segments_meet_not_at_endpoint((b, c), (u, v))
    False
    """
    a, b = ab
    u, v = uv
    if a == b or u == v:
        raise ValueError('Segment has no interior')
    if not coplanar(a, b, u, v):
        return False
    if colinear(a, b, u):
        if colinear(a, b, v):
            s = find_convex_combination(a, b, u)
            t = find_convex_combination(a, b, v)
            if s > t:
                s, t = (t, s)
            return not (t <= 0 or s >= 1)
        else:
            return point_meets_interior_of_segment(u, (a, b))
    else:
        if are_parallel(b - a, v - u):
            return False
        M = Matrix([b - a, u - v]).transpose()
        s, t = M.solve_right(u - a)
        return 0 < s < 1 and 0 <= t <= 1 or (0 <= s <= 1 and 0 < t < 1)