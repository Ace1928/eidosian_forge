import pytest
def test_stop_all(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100) + Animation(x=0)
    w = Widget()
    a.start(w)
    sleep(0.5)
    Animation.stop_all(w)
    assert no_animations_being_played()