from ..diffusion import DWIJointRicianLMMSEFilter
def test_DWIJointRicianLMMSEFilter_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = DWIJointRicianLMMSEFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value