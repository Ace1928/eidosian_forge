from mako import util
def invalidate_body(self):
    """Invalidate the cached content of the "body" method for this
        template.

        """
    self.invalidate('render_body', __M_defname='render_body')