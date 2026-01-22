from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def standardize_bend_matrix(a, b, c):
    a, c = (a - b, c - b)
    n = cross_product(a, c)
    M = Matrix([a, c, n]).transpose().inverse()
    return M