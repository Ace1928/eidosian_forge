from . import matrix
def images_of_original_generators(coordinate_object, penalties):
    """
    Given Ptolemy coordinates or cross ratio (or anything which supports
    get_manifold and methods to return matrices for short, middle, and long
    edges) and penalties (triple giving preferences to avoid short, middle,
    and long edges), give two lists of matrices that are the images and inverse
    images of the fundamental group generators of the unsimplified presentation.
    """
    M = coordinate_object.get_manifold()
    if M is None:
        raise Exception('Need to have a manifold')
    loops = compute_loops_for_generators(M, penalties=penalties)
    return ([_evaluate_path(coordinate_object, loop) for loop in loops], [_evaluate_path(coordinate_object, loop ** (-1)) for loop in loops])