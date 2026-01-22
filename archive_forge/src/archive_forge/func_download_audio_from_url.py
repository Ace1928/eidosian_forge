import os
import tempfile
from urllib.parse import urlparse
import requests
def download_audio_from_url(audio_url: str) -> str:
    """Download audio from url to local."""
    ext = audio_url.split('.')[-1]
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{ext}', delete=False) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return f.name