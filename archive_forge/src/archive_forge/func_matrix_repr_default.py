def matrix_repr_default(X):
    return "<%ix%i matrix, tc='%c'>" % (X.size[0], X.size[1], X.typecode)