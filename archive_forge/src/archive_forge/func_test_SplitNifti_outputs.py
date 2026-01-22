from ..dcmstack import SplitNifti
def test_SplitNifti_outputs():
    output_map = dict(out_list=dict())
    outputs = SplitNifti.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value