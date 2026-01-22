from ..utils import RobustFOV
def test_RobustFOV_outputs():
    output_map = dict(out_roi=dict(extensions=None), out_transform=dict(extensions=None))
    outputs = RobustFOV.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value