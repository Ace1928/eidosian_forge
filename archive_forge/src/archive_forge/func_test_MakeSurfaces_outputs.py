from ..utils import MakeSurfaces
def test_MakeSurfaces_outputs():
    output_map = dict(out_area=dict(extensions=None), out_cortex=dict(extensions=None), out_curv=dict(extensions=None), out_pial=dict(extensions=None), out_thickness=dict(extensions=None), out_white=dict(extensions=None))
    outputs = MakeSurfaces.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value