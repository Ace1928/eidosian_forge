import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def fs_cothub_gsm8k_match_answer(task_data, response):
    pattern = '\\d*\\.?\\d+'
    pred = re.findall(pattern, response)
    if len(pred) >= 1:
        return (True, pred[-1])
    return (False, response)