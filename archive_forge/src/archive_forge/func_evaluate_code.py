from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def evaluate_code(code, inputs=None, state=None, verbose=False, return_interpretor_error=False):
    tools = BASE_PYTHON_TOOLS.copy()
    for name, tool in TEST_TOOLS.items():
        if name not in code:
            continue
        tools[name] = tool
    if isinstance(inputs, dict):
        inputs = inputs.copy()
    elif inputs is not None:
        inputs = {inp: f'<<{inp}>>' for inp in inputs}
    if state is not None:
        state.update(inputs)
    else:
        state = inputs
    try:
        return evaluate(code, tools, state)
    except InterpretorError as e:
        return str(e)
    except Exception as e:
        if verbose:
            print(e)
        return None