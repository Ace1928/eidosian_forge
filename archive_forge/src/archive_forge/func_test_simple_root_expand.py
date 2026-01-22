from pyxnat import select
def test_simple_root_expand():
    assert select.compute('//experiments') == ['/projects/*/subjects/*/experiments/*']