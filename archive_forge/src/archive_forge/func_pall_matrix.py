from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def pall_matrix(a, b, c, d):
    """
    For testing, here are all 3 x 3 rational orthogonal matrices,
    which Pall parameterized by four integers.
    """
    n = a ** 2 + b ** 2 + c ** 2 + d ** 2
    assert n % 2 == 1
    entries = [[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (-a * d + b * c), 2 * (a * c + b * d)], [2 * (a * d + b * c), a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * (-a * b + c * d)], [2 * (-a * c + b * d), 2 * (a * b + c * d), a ** 2 - b ** 2 - c ** 2 + d ** 2]]
    return Matrix([[e / QQ(n) for e in row] for row in entries])