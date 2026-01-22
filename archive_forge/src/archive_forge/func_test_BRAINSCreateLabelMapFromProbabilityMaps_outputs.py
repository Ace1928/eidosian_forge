from ..specialized import BRAINSCreateLabelMapFromProbabilityMaps
def test_BRAINSCreateLabelMapFromProbabilityMaps_outputs():
    output_map = dict(cleanLabelVolume=dict(extensions=None), dirtyLabelVolume=dict(extensions=None))
    outputs = BRAINSCreateLabelMapFromProbabilityMaps.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value