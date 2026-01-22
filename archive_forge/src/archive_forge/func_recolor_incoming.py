def recolor_incoming(self, palette=None, color=None):
    """
        If this vertex lies in a non-closed component, recolor its incoming
        path.  The old color is not freed.  This vertex is NOT recolored. 
        """
    v = self
    while True:
        e = v.in_arrow
        if not e:
            break
        v = e.start
        if v == self:
            return
    if not color:
        color = palette.new()
    v = self
    while True:
        e = v.in_arrow
        if not e:
            break
        e.set_color(color)
        v = e.start
        v.set_color(color)