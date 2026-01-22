import os
import stat
from ..osutils import pathjoin
from ..trace import warning
def build_tree_contents(template):
    """Reconstitute some files from a text description.

    Each element of template is a tuple.  The first element is a filename,
    with an optional ending character indicating the type.

    The template is built relative to the Python process's current
    working directory.

    ('foo/',) will build a directory.
    ('foo', 'bar') will write 'bar' to 'foo'
    ('foo@', 'linktarget') will raise an error
    """
    for tt in template:
        name = tt[0]
        if name[-1] == '/':
            os.mkdir(name)
        elif name[-1] == '@':
            os.symlink(tt[1], tt[0][:-1])
        else:
            with open(name, 'w' + ('b' if isinstance(tt[1], bytes) else '')) as f:
                f.write(tt[1])