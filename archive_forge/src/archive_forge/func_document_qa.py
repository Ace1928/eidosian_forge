from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def document_qa(image, question):
    return f'This is the answer to {question} from the document {image}.'