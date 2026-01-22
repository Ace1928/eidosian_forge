from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
def ripple_fade(self):
    """Finish ripple animation on current widget.
        """
    Animation.cancel_all(self, 'ripple_rad', 'ripple_color')
    width, height = self.size
    rc = self.ripple_color
    duration = self.ripple_duration_out
    anim = Animation(ripple_rad=max(width, height) * self.ripple_scale, ripple_color=[rc[0], rc[1], rc[2], 0.0], t=self.ripple_func_out, duration=duration)
    anim.bind(on_complete=self._ripple_anim_complete)
    anim.start(self)