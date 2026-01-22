from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
def strip_move(self, instance, touch):
    if touch.grab_current is not instance:
        return False
    max_size = self.max_size
    min_size = self.min_size
    sz_frm = self.sizable_from[0]
    if sz_frm in ('t', 'b'):
        diff_y = touch.dy
        self_y = self.y
        self_top = self.top
        if not self._is_moving(sz_frm, diff_y, touch.y, self_y, self_top):
            return
        if self.keep_within_parent:
            if sz_frm == 't' and self_top + diff_y > self.parent.top:
                diff_y = self.parent.top - self_top
            elif sz_frm == 'b' and self_y + diff_y < self.parent.y:
                diff_y = self.parent.y - self_y
        if sz_frm == 'b':
            diff_y *= -1
        if self.size_hint_y:
            self.size_hint_y = None
        if self.height > 0:
            self.height += diff_y
        else:
            self.height = 1
        height = self.height
        self.height = max(min_size, min(height, max_size))
        self._parent_proportion = self.height / self.parent.height
    else:
        diff_x = touch.dx
        self_x = self.x
        self_right = self.right
        if not self._is_moving(sz_frm, diff_x, touch.x, self_x, self_right):
            return
        if self.keep_within_parent:
            if sz_frm == 'l' and self_x + diff_x < self.parent.x:
                diff_x = self.parent.x - self_x
            elif sz_frm == 'r' and self_right + diff_x > self.parent.right:
                diff_x = self.parent.right - self_right
        if sz_frm == 'l':
            diff_x *= -1
        if self.size_hint_x:
            self.size_hint_x = None
        if self.width > 0:
            self.width += diff_x
        else:
            self.width = 1
        width = self.width
        self.width = max(min_size, min(width, max_size))
        self._parent_proportion = self.width / self.parent.width