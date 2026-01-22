from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import List
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from .manager import lsp_message_listener
from .paths import file_uri_to_path, is_relative
from .types import LanguageServerManagerAPI
def setup_shadow_filesystem(virtual_documents_uri: str):
    if not virtual_documents_uri.startswith('file:/'):
        raise ShadowFilesystemError('Virtual documents URI has to start with "file:/", got ' + virtual_documents_uri)
    initialized = False
    failures: List[Exception] = []
    shadow_filesystem = Path(file_uri_to_path(virtual_documents_uri))

    @lsp_message_listener('client')
    async def shadow_virtual_documents(scope, message, language_server, manager):
        """Intercept a message with document contents creating a shadow file for it.

        Only create the shadow file if the URI matches the virtual documents URI.
        Returns the path on filesystem where the content was stored.
        """
        nonlocal initialized
        server_spec = manager.language_servers[language_server]
        if not server_spec.get('requires_documents_on_disk', True):
            return
        if not message.get('method') in WRITE_ONE:
            return
        document = extract_or_none(message, ['params', 'textDocument'])
        if document is None:
            raise ShadowFilesystemError('Could not get textDocument from: {}'.format(message))
        uri = extract_or_none(document, ['uri'])
        if not uri:
            raise ShadowFilesystemError('Could not get URI from: {}'.format(message))
        if not uri.startswith(virtual_documents_uri):
            return
        if not initialized:
            if len(failures) == 3:
                return
            try:
                shadow_filesystem.mkdir(parents=True, exist_ok=True)
                rmtree(str(shadow_filesystem))
                shadow_filesystem.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError, FileNotFoundError) as e:
                failures.append(e)
                if len(failures) == 3:
                    manager.log.warn('[lsp] initialization of shadow filesystem failed three times check if the path set by `LanguageServerManager.virtual_documents_dir` or `JP_LSP_VIRTUAL_DIR` is correct; if this is happening with a server for which you control (or wish to override) jupyter-lsp specification you can try switching `requires_documents_on_disk` off. The errors were: %s', failures)
                return
            initialized = True
        path = file_uri_to_path(uri)
        if not is_relative(shadow_filesystem, path):
            raise ShadowFilesystemError(f'Path {path} is not relative to shadow filesystem root')
        editable_file = EditableFile(path)
        await editable_file.read()
        text = extract_or_none(document, ['text'])
        if text is not None:
            changes = [{'text': text}]
        else:
            if message['method'] != 'textDocument/didChange':
                return
            if 'contentChanges' not in message['params']:
                raise ShadowFilesystemError('textDocument/didChange is missing contentChanges')
            changes = message['params']['contentChanges']
        if len(changes) > 1:
            manager.log.warn('LSP warning: up to one change supported for textDocument/didChange')
        for change in changes[:1]:
            change_range = change.get('range', editable_file.full_range)
            editable_file.apply_change(change['text'], **change_range)
        await editable_file.write()
        return path
    return shadow_virtual_documents