from pyxnat import Interface
def test_inspector_structure():
    from pyxnat.core import Inspector
    i = Inspector(central)
    i.set_autolearn()
    print(i.datatypes())
    i.structure()