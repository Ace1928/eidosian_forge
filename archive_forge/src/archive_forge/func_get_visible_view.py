from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
def get_visible_view(self, index):
    """Returns the currently visible view associated with ``index``.

        If no view is currently displayed for ``index`` it returns ``None``.
        """
    return self.views.get(index)