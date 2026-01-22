def test_parse_filename_do_not_split():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('user@local/file@name', do_not_split_by_at_sign=True)
    assert filename == 'user@local/file@name'
    assert index is None