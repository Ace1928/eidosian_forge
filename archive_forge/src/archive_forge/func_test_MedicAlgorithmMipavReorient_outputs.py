from ..developer import MedicAlgorithmMipavReorient
def test_MedicAlgorithmMipavReorient_outputs():
    output_map = dict()
    outputs = MedicAlgorithmMipavReorient.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value