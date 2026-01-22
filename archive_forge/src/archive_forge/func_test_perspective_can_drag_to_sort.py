from panel.pane import Perspective
def test_perspective_can_drag_to_sort():
    msg = {'sort': [['Curve']]}
    actual = Perspective()._process_property_change(msg)
    assert actual == msg