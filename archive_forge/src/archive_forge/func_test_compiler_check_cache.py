import linecache
import sys
from IPython.core import compilerop
def test_compiler_check_cache():
    """Test the compiler properly manages the cache.
    """
    cp = compilerop.CachingCompiler()
    cp.cache('x=1', 99)
    linecache.checkcache()
    assert any((k.startswith('<ipython-input-99') for k in linecache.cache)), 'Entry for input-99 missing from linecache'