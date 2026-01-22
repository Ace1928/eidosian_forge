from unittest import TestCase
from traitlets import TraitError
import ipywidgets as widgets
def test_construction_with_children(self):
    html = widgets.HTML('some html')
    slider = widgets.IntSlider()
    box = widgets.Box([html, slider])
    children_state = box.get_state()['children']
    assert children_state == [widgets.widget._widget_to_json(html, None), widgets.widget._widget_to_json(slider, None)]