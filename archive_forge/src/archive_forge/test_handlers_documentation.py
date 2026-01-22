from IPython.core import autocall
from IPython.testing import tools as tt
Loop through a list of (pre, post) inputs, where pre is the string
    handed to ipython, and post is how that string looks after it's been
    transformed (i.e. ipython's notion of _i)