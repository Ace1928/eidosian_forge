import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def dot2img(dot_string, t='svg'):
    """
    Create image representation fom dot_string, using the 'dot' program
    from the Graphviz package.

    Use the 't' argument to specify the image file format, for ex. 'jpeg', 'eps',
    'json', 'png' or 'webp' (Running 'dot -T:' lists all available formats).

    Note that the "capture_output" option of subprocess.run() is only available
    with text formats (like svg), but not with binary image formats (like png).
    """
    try:
        find_binary('dot')
        try:
            if t in ['dot', 'dot_json', 'json', 'svg']:
                proc = subprocess.run(['dot', '-T%s' % t], capture_output=True, input=dot_string, text=True)
            else:
                proc = subprocess.run(['dot', '-T%s' % t], input=bytes(dot_string, encoding='utf8'))
            return proc.stdout
        except:
            raise Exception('Cannot create image representation by running dot from string: {}'.format(dot_string))
    except OSError as e:
        raise Exception('Cannot find the dot binary from Graphviz package') from e