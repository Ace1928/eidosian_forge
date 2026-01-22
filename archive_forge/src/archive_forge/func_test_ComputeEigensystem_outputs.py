from ..dti import ComputeEigensystem
def test_ComputeEigensystem_outputs():
    output_map = dict(eigen=dict(extensions=None))
    outputs = ComputeEigensystem.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value