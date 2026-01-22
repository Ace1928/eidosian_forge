import pytest
def test_start_animation(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100, d=1)
    w = Widget()
    a.start(w)
    sleep(1.5)
    assert w.x == pytest.approx(100)
    assert no_animations_being_played()