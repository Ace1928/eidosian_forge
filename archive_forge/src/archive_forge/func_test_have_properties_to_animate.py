import pytest
def test_have_properties_to_animate(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100) & Animation(y=100)
    w = Widget()
    assert not a.have_properties_to_animate(w)
    a.start(w)
    assert a.have_properties_to_animate(w)
    a.stop(w)
    assert not a.have_properties_to_animate(w)
    assert no_animations_being_played()