from ..model import FEATModel
def test_FEATModel_outputs():
    output_map = dict(con_file=dict(extensions=None), design_cov=dict(extensions=None), design_file=dict(extensions=None), design_image=dict(extensions=None), fcon_file=dict(extensions=None))
    outputs = FEATModel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value