from ..io import BIDSDataGrabber
def test_BIDSDataGrabber_inputs():
    input_map = dict(base_dir=dict(mandatory=True), extra_derivatives=dict(), index_derivatives=dict(mandatory=True, usedefault=True), load_layout=dict(mandatory=False), output_query=dict(), raise_on_empty=dict(usedefault=True))
    inputs = BIDSDataGrabber.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value