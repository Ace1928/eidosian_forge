def test_parse_filename_with_at_no_ext():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('path.to/filename@1:4')
    assert filename == 'path.to/filename'
    assert index == slice(1, 4, None)