def setIconTheme(theme):
    global icon_theme
    icon_theme = theme
    import xdg.IconTheme
    xdg.IconTheme.themes = []