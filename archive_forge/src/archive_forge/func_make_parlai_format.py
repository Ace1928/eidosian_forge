from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
import json
def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for each in data:
            output = []
            story = each['story'].replace('\n', '\\n')
            for question, ans in zip(each['questions'], each['answers']):
                question_txt = ''
                if question['turn_id'] == 1:
                    question_txt = story + '\\n' + question['input_text']
                else:
                    question_txt = question['input_text']
                output.append('text:{question}\tlabels:{labels}'.format(question=question_txt, labels=ans['input_text'].replace('|', ' __PIPE__ ')))
                if question['turn_id'] < len(each['questions']):
                    output.append('\n')
            output.append('\t\tepisode_done:True\n')
            fout.write(''.join(output))