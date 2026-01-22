from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def enable_client_permessage_deflate(extensions: Optional[Sequence[ClientExtensionFactory]]) -> Sequence[ClientExtensionFactory]:
    """
    Enable Per-Message Deflate with default settings in client extensions.

    If the extension is already present, perhaps with non-default settings,
    the configuration isn't changed.

    """
    if extensions is None:
        extensions = []
    if not any((extension_factory.name == ClientPerMessageDeflateFactory.name for extension_factory in extensions)):
        extensions = list(extensions) + [ClientPerMessageDeflateFactory(compress_settings={'memLevel': 5})]
    return extensions