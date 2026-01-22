from ..csv import CSVReader
def test_CSVReader_inputs():
    input_map = dict(header=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True))
    inputs = CSVReader.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value