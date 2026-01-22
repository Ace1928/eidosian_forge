from ..tracks import StreamlineTractography
def test_StreamlineTractography_outputs():
    output_map = dict(gfa=dict(extensions=None), odf_peaks=dict(extensions=None), out_seeds=dict(extensions=None), tracks=dict(extensions=None))
    outputs = StreamlineTractography.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value