from typing import Dict
import torch
from torch._guards import detect_fake_mode
from torch.export.exported_program import InputKind, InputSpec, TensorArgument

    Takes an ExportedProgram and returns the ExportedProgram modified in-place,
    with the constant tensors as buffers.
    