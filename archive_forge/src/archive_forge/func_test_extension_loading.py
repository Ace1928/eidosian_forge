import os.path
from tempfile import TemporaryDirectory
import IPython.testing.tools as tt
from IPython.utils.syspathcontext import prepended_to_syspath
def test_extension_loading():
    em = get_ipython().extension_manager
    with TemporaryDirectory() as td:
        ext1 = os.path.join(td, 'ext1.py')
        with open(ext1, 'w', encoding='utf-8') as f:
            f.write(ext1_content)
        ext2 = os.path.join(td, 'ext2.py')
        with open(ext2, 'w', encoding='utf-8') as f:
            f.write(ext2_content)
        with prepended_to_syspath(td):
            assert 'ext1' not in em.loaded
            assert 'ext2' not in em.loaded
            with tt.AssertPrints('Running ext1 load'):
                assert em.load_extension('ext1') is None
            assert 'ext1' in em.loaded
            with tt.AssertNotPrints('Running ext1 load'):
                assert em.load_extension('ext1') == 'already loaded'
            with tt.AssertPrints('Running ext1 unload'):
                with tt.AssertPrints('Running ext1 load', suppress=False):
                    em.reload_extension('ext1')
            with tt.AssertPrints('Running ext1 unload'):
                assert em.unload_extension('ext1') is None
            with tt.AssertNotPrints('Running ext1 unload'):
                assert em.unload_extension('ext1') == 'not loaded'
            assert em.unload_extension('ext2') == 'not loaded'
            with tt.AssertPrints('Running ext2 load'):
                assert em.load_extension('ext2') is None
            assert em.unload_extension('ext2') == 'no unload function'
            with tt.AssertPrints('Running ext2 load'):
                em.reload_extension('ext2')