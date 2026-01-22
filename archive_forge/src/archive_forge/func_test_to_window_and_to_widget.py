import pytest
import functools
@pytest.mark.parametrize('root_widget_cls_name', all_widget_cls_names)
@pytest.mark.parametrize('target_widget_cls_name', all_widget_cls_names)
def test_to_window_and_to_widget(root_widget_cls_name, target_widget_cls_name, kivy_clock):
    from textwrap import dedent
    from kivy.lang import Builder
    root = Builder.load_string(dedent("\n        {}:\n            pos: 100, 0\n\n            # In case the root widget is ScrollView, this cushion is needed,\n            # because ScrollView's direct child is always at pos(0, 0)\n            Widget:\n                pos: 0, 0\n\n                {}:\n                    id: target\n                    pos: 0, 100\n        ").format(root_widget_cls_name, target_widget_cls_name))
    kivy_clock.tick()
    target = root.ids.target
    if is_relative_type(root):
        assert target.to_window(*target.pos) == (100, 100)
        assert target.to_widget(0, 0) == ((-100, -100) if is_relative_type(target) else (-100, 0))
    else:
        assert target.to_window(*target.pos) == (0, 100)
        assert target.to_widget(0, 0) == ((0, -100) if is_relative_type(target) else (0, 0))