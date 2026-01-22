from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
Translates input text and returns translated text.

      Args:
        request: (TranslateProjectsTranslateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TranslateTextResponse) The response message.
      