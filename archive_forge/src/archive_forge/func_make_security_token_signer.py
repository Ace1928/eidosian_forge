from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
def make_security_token_signer(oci_config):
    pk = oci.signer.load_private_key_from_file(oci_config.get('key_file'), None)
    with open(oci_config.get('security_token_file'), encoding='utf-8') as f:
        st_string = f.read()
    return oci.auth.signers.SecurityTokenSigner(st_string, pk)