from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def schema_builder(self, templ: dict, dimension: int) -> str:
    metadata_proto_tpl = '\n/**\n* @Indexed\n*/\nmessage %s {\n/**\n* @Vector(dimension=%d)\n*/\nrepeated float %s = 1;\n'
    metadata_proto = metadata_proto_tpl % (self._entity_name, dimension, self._vectorfield)
    idx = 2
    for f, v in templ.items():
        if isinstance(v, str):
            metadata_proto += 'optional string ' + f + ' = ' + str(idx) + ';\n'
        elif isinstance(v, int):
            metadata_proto += 'optional int64 ' + f + ' = ' + str(idx) + ';\n'
        elif isinstance(v, float):
            metadata_proto += 'optional double ' + f + ' = ' + str(idx) + ';\n'
        elif isinstance(v, bytes):
            metadata_proto += 'optional bytes ' + f + ' = ' + str(idx) + ';\n'
        elif isinstance(v, bool):
            metadata_proto += 'optional bool ' + f + ' = ' + str(idx) + ';\n'
        else:
            raise Exception('Unable to build proto schema for metadata. Unhandled type for field: ' + f)
        idx += 1
    metadata_proto += '}\n'
    return metadata_proto