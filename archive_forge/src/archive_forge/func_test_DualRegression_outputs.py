from ..model import DualRegression
def test_DualRegression_outputs():
    output_map = dict(out_dir=dict())
    outputs = DualRegression.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value