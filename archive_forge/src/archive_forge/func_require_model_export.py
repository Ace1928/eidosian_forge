from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Field
def require_model_export(model_id: str, revision: Any=None, subfolder: Any=None) -> bool:
    model_dir = Path(model_id)
    if subfolder is not None:
        model_dir = model_dir / subfolder
    if model_dir.is_dir():
        return not (model_dir / 'openvino_model.xml').exists() or not (model_dir / 'openvino_model.bin').exists()
    hf_api = HfApi()
    try:
        model_info = hf_api.model_info(model_id, revision=revision or 'main')
        normalized_subfolder = None if subfolder is None else Path(subfolder).as_posix()
        model_files = [file.rfilename for file in model_info.siblings if normalized_subfolder is None or file.rfilename.startswith(normalized_subfolder)]
        ov_model_path = 'openvino_model.xml' if subfolder is None else f'{normalized_subfolder}/openvino_model.xml'
        return ov_model_path not in model_files or ov_model_path.replace('.xml', '.bin') not in model_files
    except Exception:
        return True