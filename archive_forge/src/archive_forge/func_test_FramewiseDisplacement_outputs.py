from ..confounds import FramewiseDisplacement
def test_FramewiseDisplacement_outputs():
    output_map = dict(fd_average=dict(), out_figure=dict(extensions=None), out_file=dict(extensions=None))
    outputs = FramewiseDisplacement.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value