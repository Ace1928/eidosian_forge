from ..dti import BEDPOSTX5
def test_BEDPOSTX5_outputs():
    output_map = dict(dyads=dict(), dyads_dispersion=dict(), mean_S0samples=dict(extensions=None), mean_dsamples=dict(extensions=None), mean_fsamples=dict(), mean_phsamples=dict(), mean_thsamples=dict(), merged_fsamples=dict(), merged_phsamples=dict(), merged_thsamples=dict())
    outputs = BEDPOSTX5.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value