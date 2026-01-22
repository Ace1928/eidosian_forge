import os
def test_default_virtual_documents_dir(app):
    app.initialize(["--ServerApp.jpserver_extensions={'jupyter_lsp.serverextension': True}"])
    assert app.language_server_manager.virtual_documents_dir == '.virtual_documents'