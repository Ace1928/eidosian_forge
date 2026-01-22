from pyglet import gl
from pyglet import app
from pyglet import window
from pyglet import canvas
def get_closest_mode(self, width, height):
    """Get the screen mode that best matches a given size.

        If no supported mode exactly equals the requested size, a larger one
        is returned; or ``None`` if no mode is large enough.

        :Parameters:
            `width` : int
                Requested screen width.
            `height` : int
                Requested screen height.

        :rtype: :class:`ScreenMode`

        .. versionadded:: 1.2
        """
    current = self.get_mode()
    best = None
    for mode in self.get_modes():
        if mode.width < width or mode.height < height:
            continue
        if best is None:
            best = mode
        if mode.width <= best.width and mode.height <= best.height and (mode.width < best.width or mode.height < best.height):
            best = mode
        if mode.width == best.width and mode.height == best.height:
            points = 0
            if mode.rate == current.rate:
                points += 2
            if best.rate == current.rate:
                points -= 2
            if mode.depth == current.depth:
                points += 1
            if best.depth == current.depth:
                points -= 1
            if points > 0:
                best = mode
    return best