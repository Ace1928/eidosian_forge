import sys, re
def getstyle(self, tag):
    """ return attribute list suitable for styling. """
    try:
        styledict = tag.style.__dict__
    except AttributeError:
        return []
    else:
        stylelist = [x + ': ' + y for x, y in styledict.items()]
        return [u(' style="%s"') % u('; ').join(stylelist)]