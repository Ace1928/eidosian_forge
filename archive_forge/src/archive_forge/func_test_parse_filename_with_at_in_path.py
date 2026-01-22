def test_parse_filename_with_at_in_path():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('user@local/filename.xyz')
    assert filename == 'user@local/filename.xyz'
    assert index is None