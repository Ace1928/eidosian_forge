import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def project_to_diagram(link_in_R3):
    """
    >>> project_to_diagram(fig8_points())
    <Link: 1 comp; 4 cross>
    """
    diagram = None
    for mat_size in [None, 15, 25, 15, 25, 15, 25, 100, 250, 100, 250, 500, 500]:
        if mat_size is None:
            proj_mat = Matrix([[3, 1, 0], [1, -1, 5], [0, 0, -1]])
        else:
            proj_mat = random_transform(mat_size)
        try:
            projection = LinkProjection(link_in_R3, proj_mat)
            diagram = projection.link()
            break
        except GeneralPositionError as e:
            pass
    return diagram