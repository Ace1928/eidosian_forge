from __future__ import annotations
from typing import Any, ItemsView, Iterator, Mapping, MutableMapping, Optional
from bson import _get_object_size, _raw_to_dict
from bson.codec_options import _RAW_BSON_DOCUMENT_MARKER, CodecOptions
from bson.codec_options import DEFAULT_CODEC_OPTIONS as DEFAULT
from bson.son import SON
Lazily decode and iterate elements in this document.