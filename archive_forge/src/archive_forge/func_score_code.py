from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def score_code(agent_answer, theoretical_answer, verbose: bool=False):
    if verbose:
        print(agent_answer, theoretical_answer)
    theoretical_answer = theoretical_answer if isinstance(theoretical_answer, list) else [theoretical_answer]
    if agent_answer in theoretical_answer:
        if verbose:
            print('Perfect!')
        return 1
    elif isinstance(agent_answer, dict) and any((v in theoretical_answer for v in agent_answer.values())):
        if verbose:
            print('Almsot perfect, result in state!')
        return 0.75
    else:
        if verbose:
            print('Result is not the right one but code executed.')
        return 0.3