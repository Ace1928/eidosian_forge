from typing import Any, Dict, Optional
from PIL import Image
from gradio import components
def is_diffusers_pipeline_type(pipeline, class_name: str):
    cls = getattr(diffusers, class_name, None)
    return cls and isinstance(pipeline, cls)