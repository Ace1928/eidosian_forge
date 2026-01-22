from parlai.core.teachers import DialogTeacher
from .build import build
import os
import copy
def convert_to_qa(input_data):
    lines = input_data.split('\n')
    context = lines[1]
    predicate_count = int(lines[0].split('\t')[-1])
    unparsed_qa = lines[2:]

    def parse_qa(qa_line):
        qa_split = qa_line.split('\t?\t')
        question = context + '\n' + qa_split[0].replace('\t_', '').replace('\t', ' ') + '?'
        answers = qa_split[1].split(' ### ')
        return [question, answers]
    qa_pairs = []
    counter = 0
    for _i in range(predicate_count):
        question_count = int(unparsed_qa[counter].split('\t')[-1])
        counter += 1
        for _j in range(question_count):
            qa_pairs.append(parse_qa(unparsed_qa[counter]))
            counter += 1
    return qa_pairs