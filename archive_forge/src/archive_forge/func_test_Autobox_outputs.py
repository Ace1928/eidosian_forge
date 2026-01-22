from ..utils import Autobox
def test_Autobox_outputs():
    output_map = dict(out_file=dict(extensions=None), x_max=dict(), x_min=dict(), y_max=dict(), y_min=dict(), z_max=dict(), z_min=dict())
    outputs = Autobox.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value