import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def authors_from_one_paragraph(self, field):
    """Return list of Text nodes for ";"- or ","-separated authornames."""
    text = ''.join((str(node) for node in field[1].traverse(nodes.Text)))
    if not text:
        raise TransformError
    for authorsep in self.language.author_separators:
        pattern = '(?<!\x00)%s' % authorsep
        authornames = re.split(pattern, text)
        if len(authornames) > 1:
            break
    authornames = (name.strip() for name in authornames)
    authors = [[nodes.Text(name, utils.unescape(name, True))] for name in authornames if name]
    return authors