from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def image_captioner(image):
    if 'image' not in image:
        raise ValueError(f'`image` ({image}) is not an image.')
    return f'This is a description of {image}.'