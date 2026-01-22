from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
def compute_binarizer(x, threshold=None):
    return ((x > threshold).astype(x.dtype),)