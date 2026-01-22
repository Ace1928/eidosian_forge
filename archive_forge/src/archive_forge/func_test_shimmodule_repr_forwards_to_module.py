from IPython.utils.shimmodule import ShimModule
import IPython
def test_shimmodule_repr_forwards_to_module():
    shim_module = ShimModule('shim_module', mirror='IPython')
    assert repr(shim_module) == repr(IPython)