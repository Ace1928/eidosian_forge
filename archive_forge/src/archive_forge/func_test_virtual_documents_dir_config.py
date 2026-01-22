import os
def test_virtual_documents_dir_config(app):
    custom_dir = '.custom_virtual_dir'
    app.initialize(["--ServerApp.jpserver_extensions={'jupyter_lsp.serverextension': True}", '--ServerApp.LanguageServerManager.virtual_documents_dir=' + custom_dir])
    assert app.language_server_manager.virtual_documents_dir == custom_dir