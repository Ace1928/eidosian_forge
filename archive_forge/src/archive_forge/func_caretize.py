def caretize(self, indent=0):
    """Produces a user-readable two line string indicating the origin of
        some code. Example::

          y ~ x1:x2
              ^^

        If optional argument 'indent' is given, then both lines will be
        indented by this much. The returned string does not have a trailing
        newline.
        """
    return '%s%s\n%s%s%s' % (' ' * indent, self.code, ' ' * indent, ' ' * self.start, '^' * (self.end - self.start))