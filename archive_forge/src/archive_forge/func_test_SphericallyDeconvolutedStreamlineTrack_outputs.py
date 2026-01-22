from ..tracking import SphericallyDeconvolutedStreamlineTrack
def test_SphericallyDeconvolutedStreamlineTrack_outputs():
    output_map = dict(tracked=dict(extensions=None))
    outputs = SphericallyDeconvolutedStreamlineTrack.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value