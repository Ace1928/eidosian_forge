from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def evaluate_agent(agent, batch_size=8, verbose=False, return_errors=False):
    """
    Evaluates a new agent on all `EVALUATION_TASKS`.

    Example:

    ```py
    agent = NewOpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = new_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    agent_tools = set(agent.toolbox.keys())
    if agent_tools != set(TEST_TOOLS):
        missing_tools = set(TEST_TOOLS) - agent_tools
        unexpected_tools = set(agent_tools) - TEST_TOOLS
        raise ValueError(f'Fix the test tools in the evaluate_agent module. Tools mising: {missing_tools}. Extra tools: {unexpected_tools}.')
    eval_tasks = []
    eval_idx = []
    for idx, pb in enumerate(EVALUATION_TASKS):
        if isinstance(pb.task, list):
            eval_tasks.extend(pb.task)
            eval_idx.extend([idx] * len(pb.task))
        else:
            eval_tasks.append(pb.task)
            eval_idx.append(idx)
    tool_selection_score = 0
    tool_used_score = 0
    code_score = 0
    if return_errors:
        tool_selection_errors = {}
        tool_used_errors = {}
        code_errors = {}
    for start_idx in range(0, len(eval_tasks), batch_size):
        end_idx = min(start_idx + batch_size, len(eval_tasks))
        batch_tasks = eval_tasks[start_idx:end_idx]
        prompts = [agent.format_prompt(task) for task in batch_tasks]
        results = agent.generate_many(prompts, stop=['Task:'])
        for idx, result in enumerate(results):
            problem = EVALUATION_TASKS[eval_idx[start_idx + idx]]
            if verbose:
                print(f'====Task {start_idx + idx}====\n{batch_tasks[idx]}\n')
            explanation, code = agent.clean_code_for_run(result)
            agent_answer = evaluate_code(code, problem.inputs, verbose=verbose)
            if isinstance(problem.answer, list):
                theoretical_answer = [evaluate_code(answer, problem.inputs) for answer in problem.answer]
            else:
                theoretical_answer = evaluate_code(problem.answer, problem.inputs)
            scores, errors = evaluate_one_result(explanation, code, agent_answer, theoretical_answer, problem.answer, verbose=verbose)
            tool_selection_score += scores[0]
            tool_used_score += scores[1]
            code_score += scores[2]
            if return_errors:
                if errors[0] is not None:
                    tool_selection_errors[batch_tasks[idx]] = errors[0]
                if errors[1] is not None:
                    tool_used_errors[batch_tasks[idx]] = errors[1]
                if errors[2] is not None:
                    code_errors[batch_tasks[idx]] = errors[2]
    scores = {'tool selection score': 100 * (tool_selection_score / len(eval_tasks)), 'tool used score': 100 * (tool_used_score / len(eval_tasks)), 'code score': 100 * (code_score / len(eval_tasks))}
    if return_errors:
        return (scores, tool_selection_errors, tool_used_errors, code_errors)
    else:
        return scores