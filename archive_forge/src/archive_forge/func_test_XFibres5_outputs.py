from ..dti import XFibres5
def test_XFibres5_outputs():
    output_map = dict(dyads=dict(), fsamples=dict(), mean_S0samples=dict(extensions=None), mean_dsamples=dict(extensions=None), mean_fsamples=dict(), mean_tausamples=dict(extensions=None), phsamples=dict(), thsamples=dict())
    outputs = XFibres5.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value