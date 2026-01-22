from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def speaker(text):
    return f'This is actually a sound reading {text}.'