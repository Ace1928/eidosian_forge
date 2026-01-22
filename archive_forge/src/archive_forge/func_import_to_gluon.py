from .import_onnx import GraphProto
def import_to_gluon(model_file, ctx):
    """
    Imports the ONNX model files, passed as a parameter, into Gluon SymbolBlock object.

    Parameters
    ----------
    model_file : str
        ONNX model file name
    ctx : Context or list of Context
        Loads the model into one or many context(s).

    Returns
    -------
    sym_block : :class:`~mxnet.gluon.SymbolBlock`
        A SymbolBlock object representing the given model file.

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    """
    graph = GraphProto()
    try:
        import onnx
    except ImportError:
        raise ImportError('Onnx and protobuf need to be installed. Instructions to' + ' install - https://github.com/onnx/onnx#installation')
    model_proto = onnx.load_model(model_file)
    model_opset_version = max([x.version for x in model_proto.opset_import])
    net = graph.graph_to_gluon(model_proto.graph, ctx, model_opset_version)
    return net