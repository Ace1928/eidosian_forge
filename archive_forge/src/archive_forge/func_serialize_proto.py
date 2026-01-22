import os
import tempfile
import unittest
import onnx
def serialize_proto(self, proto) -> bytes:
    text = onnx.printer.to_text(proto)
    return text.encode('utf-8')