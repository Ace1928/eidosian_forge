from ..brains import fcsv_to_hdf5
def test_fcsv_to_hdf5_outputs():
    output_map = dict(landmarksInformationFile=dict(extensions=None), modelFile=dict(extensions=None))
    outputs = fcsv_to_hdf5.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value