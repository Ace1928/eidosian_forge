from snappy.ptolemy.homology import homology_basis_representatives_with_orders
def face_var_name_to_index(var_name):
    """
    Convert variable name to index in array of weights.

        >>> face_var_name_to_index('s_2_5')
        22

    """
    name, face_index, tet_index = var_name.split('_')
    if name != 's':
        raise AssertionError("Variable name '%s' for face class invalid" % var_name)
    return 4 * int(tet_index) + int(face_index)