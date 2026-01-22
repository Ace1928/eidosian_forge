from ..base import WatershedBEM
def test_WatershedBEM_outputs():
    output_map = dict(brain_surface=dict(extensions=None, loc='bem/watershed'), cor_files=dict(altkey='COR', loc='bem/watershed/ws'), fif_file=dict(altkey='fif', extensions=None, loc='bem'), inner_skull_surface=dict(extensions=None, loc='bem/watershed'), mesh_files=dict(), outer_skin_surface=dict(extensions=None, loc='bem/watershed'), outer_skull_surface=dict(extensions=None, loc='bem/watershed'))
    outputs = WatershedBEM.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value