from Xlib.X import NoSymbol
def load_keysym_group(group):
    """Load all the keysyms in group.

    Given a group name such as 'latin1' or 'katakana' load the keysyms
    defined in module 'Xlib.keysymdef.group-name' into this XK module."""
    if '.' in group:
        raise ValueError('invalid keysym group name: %s' % group)
    G = globals()
    mod = __import__('Xlib.keysymdef.%s' % group, G, locals(), [group])
    keysyms = [n for n in dir(mod) if n.startswith('XK_')]
    for keysym in keysyms:
        G[keysym] = mod.__dict__[keysym]
    del mod