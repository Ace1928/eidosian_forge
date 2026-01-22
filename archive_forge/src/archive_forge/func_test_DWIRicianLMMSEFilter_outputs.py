from ..diffusion import DWIRicianLMMSEFilter
def test_DWIRicianLMMSEFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = DWIRicianLMMSEFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value