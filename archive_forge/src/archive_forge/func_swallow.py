def swallow(self, other, palette):
    """
        Join two paths.  Self and other must be endpoints. Other is erased.
        """
    if not self.is_endpoint() or not other.is_endpoint():
        raise ValueError
    if self.in_arrow is not None:
        if other.in_arrow is not None:
            other.reverse_path()
        if self.color != other.color:
            palette.recycle(self.color)
            self.color = other.color
            self.recolor_incoming(color=other.color)
        self.out_arrow = other.out_arrow
        self.out_arrow.set_start(self)
    elif self.out_arrow is not None:
        if other.out_arrow is not None:
            other.reverse_path()
        if self.color != other.color:
            palette.recycle(other.color)
            other.recolor_incoming(color=self.color)
        self.in_arrow = other.in_arrow
        self.in_arrow.set_end(self)
    other.erase()