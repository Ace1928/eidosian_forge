import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def zs_math_match_answer(task_data, response):

    def _last_boxed_only_string(string):
        idx = string.rfind('\\boxed')
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None
        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if left_brace_idx is None or right_brace_idx is None:
            return None
        return string[left_brace_idx + 1:right_brace_idx].strip()
    ground_truth_answer = _last_boxed_only_string(task_data['_metadata']['solution'])
    assert ground_truth_answer
    response = response.strip()
    is_matched = False
    ans_marker = 'answer is'
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.startswith(':'):
            response = response[1:]
        if response.endswith('.'):
            response = response[:-1]
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed
    return (is_matched, grade_answer(response, ground_truth_answer))