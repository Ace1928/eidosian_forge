from ..dti import ProbTrackX
def test_ProbTrackX_outputs():
    output_map = dict(fdt_paths=dict(), log=dict(extensions=None), particle_files=dict(), targets=dict(), way_total=dict(extensions=None))
    outputs = ProbTrackX.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value