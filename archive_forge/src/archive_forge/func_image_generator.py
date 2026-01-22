from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def image_generator(prompt):
    return f'This is actually an image representing {prompt}.'