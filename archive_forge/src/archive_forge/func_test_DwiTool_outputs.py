from ..dwi import DwiTool
def test_DwiTool_outputs():
    output_map = dict(famap_file=dict(extensions=None), logdti_file=dict(extensions=None), mcmap_file=dict(extensions=None), mdmap_file=dict(extensions=None), rgbmap_file=dict(extensions=None), syn_file=dict(extensions=None), v1map_file=dict(extensions=None))
    outputs = DwiTool.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value