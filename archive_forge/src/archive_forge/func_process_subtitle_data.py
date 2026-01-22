from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
def process_subtitle_data(coordinates: str, data: str | bytes | Path | io.BytesIO, label: str) -> str:
    if isinstance(data, (str, Path)):
        subtitle_data = _handle_string_or_path_data(data)
    elif isinstance(data, io.BytesIO):
        subtitle_data = _handle_stream_data(data)
    elif isinstance(data, bytes):
        subtitle_data = _handle_bytes_data(data)
    else:
        raise TypeError(f'Invalid binary data format for subtitle: {type(data)}.')
    if runtime.exists():
        filename = hashlib.md5(label.encode()).hexdigest()
        file_url = runtime.get_instance().media_file_mgr.add(path_or_data=subtitle_data, mimetype='text/vtt', coordinates=coordinates, file_name=f'{filename}.vtt')
        caching.save_media_data(subtitle_data, 'text/vtt', coordinates)
        return file_url
    else:
        return ''