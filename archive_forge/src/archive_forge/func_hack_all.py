def hack_all(email=True, select=True):
    """Apply all Python 2.6 patches."""
    if email:
        hack_email()
    if select:
        hack_select()