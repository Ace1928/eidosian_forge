import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def authors_from_bullet_list(self, field):
    authors = []
    for item in field[1][0]:
        if isinstance(item, nodes.comment):
            continue
        if len(item) != 1 or not isinstance(item[0], nodes.paragraph):
            raise TransformError
        authors.append(item[0].children)
    if not authors:
        raise TransformError
    return authors