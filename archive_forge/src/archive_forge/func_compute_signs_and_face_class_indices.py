from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def compute_signs_and_face_class_indices(trig):
    """
    An array helpful to convert weights per face class to weights per
    face per tetrahedron. The entries are per face per tetrahedron and
    are a pair (sign adjustment, face class index).
    """
    result = 4 * trig.num_tetrahedra() * [None]
    face_classes = trig._ptolemy_equations_identified_face_classes()
    for i, face_class in enumerate(face_classes):
        sgn, power, repr0, repr1 = face_class
        result[face_var_name_to_index(repr0)] = (+1, i)
        result[face_var_name_to_index(repr1)] = (sgn, i)
    return result