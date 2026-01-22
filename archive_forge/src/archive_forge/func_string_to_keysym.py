from Xlib.X import NoSymbol
def string_to_keysym(keysym):
    """Return the (16 bit) numeric code of keysym.

    Given the name of a keysym as a string, return its numeric code.
    Don't include the 'XK_' prefix, just use the base, i.e. 'Delete'
    instead of 'XK_Delete'."""
    return globals().get('XK_' + keysym, NoSymbol)