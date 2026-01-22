from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def polished_tetrahedra_shapes(manifold, dec_prec=None, bits_prec=200, ignore_solution_type=False):
    """
    Refines the current solution to the gluing equations to one with
    the specified accuracy.
    """
    if dec_prec is None:
        dec_prec = prec_bits_to_dec(bits_prec)
    else:
        bits_prec = prec_dec_to_bits(dec_prec)
    working_prec = dec_prec + 10
    target_espilon = float_to_pari(10.0, working_prec) ** (-dec_prec)
    if _within_sage:
        CC = ComplexField(bits_prec)
        number = CC
    else:

        def number(z):
            return Number(z, precision=bits_prec)
    if 'polished_shapes' in manifold._cache:
        curr_bits_prec, curr_sol = manifold._cache['polished_shapes']
        if bits_prec <= curr_bits_prec:
            return [number(s) for s in pari_vector_to_list(curr_sol)]
    if not ignore_solution_type and manifold.solution_type() not in ['all tetrahedra positively oriented', 'contains negatively oriented tetrahedra']:
        raise ValueError('Initial solution to gluing equations has flat or degenerate tetrahedra')
    init_shapes = pari_column_vector([complex_to_pari(complex(z), working_prec) for z in manifold.tetrahedra_shapes('rect')])
    init_equations = manifold.gluing_equations('rect')
    if gluing_equation_error(init_equations, init_shapes) > pari(1e-06):
        raise ValueError('Initial solution not very good')
    eqns = enough_gluing_equations(manifold)
    shapes = init_shapes
    initial_error = infinity_norm(gluing_equation_errors(eqns, shapes))
    for i in range(100):
        errors = gluing_equation_errors(eqns, shapes)
        error = infinity_norm(errors)
        if error < target_espilon or error > 100 * initial_error:
            break
        derivative = pari_matrix([[eqn[0][i] / z - eqn[1][i] / (1 - z) for i, z in enumerate(pari_vector_to_list(shapes))] for eqn in eqns])
        gauss = derivative.matsolve(pari_column_vector(errors))
        shapes = shapes - gauss
    error = gluing_equation_error(init_equations, shapes)
    total_change = infinity_norm(init_shapes - shapes)
    if error > 1000 * target_espilon or total_change > pari(1e-07):
        raise ValueError('Did not find a good solution to the gluing equations')
    manifold._cache['polished_shapes'] = (bits_prec, shapes)
    return [number(s) for s in pari_vector_to_list(shapes)]