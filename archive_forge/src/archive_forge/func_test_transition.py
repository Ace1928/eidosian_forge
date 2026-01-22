import pytest
def test_transition(self):
    from kivy.animation import Animation
    a = Animation(x=100) & Animation(y=100)
    with pytest.raises(AttributeError):
        a.transition