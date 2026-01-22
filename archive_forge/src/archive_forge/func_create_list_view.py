import gtk
def create_list_view(col_label):
    list_ = gtk.ListStore(str)
    list_view = gtk.TreeView(model=list_)
    list_view.set_rules_hint(True)
    list_view.get_selection().set_mode(gtk.SELECTION_SINGLE)
    renderer = gtk.CellRendererText()
    renderer.set_data('column', COLUMN_SUGGESTION)
    column = gtk.TreeViewColumn(col_label, renderer, text=COLUMN_SUGGESTION)
    list_view.append_column(column)
    return list_view