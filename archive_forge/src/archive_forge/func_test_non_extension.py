import os.path
from tempfile import TemporaryDirectory
import IPython.testing.tools as tt
from IPython.utils.syspathcontext import prepended_to_syspath
def test_non_extension():
    em = get_ipython().extension_manager
    assert em.load_extension('sys') == 'no load function'