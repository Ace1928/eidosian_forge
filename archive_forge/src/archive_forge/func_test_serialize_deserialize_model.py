import os
import tempfile
import unittest
import onnx
def test_serialize_deserialize_model(self) -> None:
    serializer = _OnnxTestTextualSerializer()
    model = onnx.parser.parse_model(_TEST_MODEL)
    serialized = serializer.serialize_proto(model)
    deserialized = serializer.deserialize_proto(serialized, onnx.ModelProto())
    self.assertEqual(model.SerializeToString(deterministic=True), deserialized.SerializeToString(deterministic=True))