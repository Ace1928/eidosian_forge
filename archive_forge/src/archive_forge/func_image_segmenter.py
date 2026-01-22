from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def image_segmenter(image, prompt):
    return f'This is the mask of {prompt} in {image}'