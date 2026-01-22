from typing import Any, Optional, Sequence
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_community.utilities.vertexai import get_client_info
Translate text documents using Google Translate.

        Arguments:
            source_language_code: ISO 639 language code of the input document.
            target_language_code: ISO 639 language code of the output document.
                For supported languages, refer to:
                https://cloud.google.com/translate/docs/languages
            mime_type: (Optional) Media Type of input text.
                Options: `text/plain`, `text/html`
        