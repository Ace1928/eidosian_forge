from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def check_face_class_weights_valid(trig, weights):
    """
    Given a SnapPy triangulation and weights per face class, check they
    form a 2-cocycle.
    """
    matrix, edge_labels, face_labels = trig._ptolemy_equations_boundary_map_2()
    for row in matrix:
        total = sum((entry * weight for entry, weight in zip(row, weights)))
        if abs(total) > 1e-06:
            raise ValueError('Weights are not a 2-cocycle.')