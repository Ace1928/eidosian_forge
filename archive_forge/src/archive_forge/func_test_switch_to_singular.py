from pyxnat import select
def test_switch_to_singular():
    assert select.compute('/projects/nosetests') == ['/project/nosetests']