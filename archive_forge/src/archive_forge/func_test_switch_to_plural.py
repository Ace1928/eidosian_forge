from pyxnat import select
def test_switch_to_plural():
    assert select.compute('/project') == ['/projects/*']