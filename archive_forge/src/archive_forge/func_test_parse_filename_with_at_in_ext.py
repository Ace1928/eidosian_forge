def test_parse_filename_with_at_in_ext():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('file_name.traj@1:4:2')
    assert filename == 'file_name.traj'
    assert index == slice(1, 4, 2)