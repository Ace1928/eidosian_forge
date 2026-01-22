import pytest
def test_animation_duration_0(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100, d=0)
    w = Widget()
    a.start(w)
    sleep(0.5)
    assert no_animations_being_played()