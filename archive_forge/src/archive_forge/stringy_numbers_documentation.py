from proto.primitives import ProtoType
A marshal between certain numeric types and strings

    This is a necessary hack to allow round trip conversion
    from messages to dicts back to messages.

    See https://github.com/protocolbuffers/protobuf/issues/2679
    and
    https://developers.google.com/protocol-buffers/docs/proto3#json
    for more details.
    