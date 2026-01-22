import inspect
from unittest import TestCase
from traitlets import TraitError
from ipywidgets import Dropdown, SelectionSlider, Select
def test_index_trigger(self):
    select = Select(options=[1, 2, 3])
    observations = []

    def f(change):
        observations.append(change.new)
    select.observe(f, 'index')
    assert select.index == 0
    select.options = [4, 5, 6]
    assert select.index == 0
    assert select.value == 4
    assert select.label == '4'
    assert observations == [0]