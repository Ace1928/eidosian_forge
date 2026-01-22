from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
def get_knowledge(self, data):
    ktypes = self.opt['knowledge_types'].split(',')
    if 'full' in ktypes or len(ktypes) >= 4:
        return data['full']
    elif 'none' in ktypes:
        return ''
    else:
        data = data['all_documents']
        ktype_order = {'plot': 0, 'review': 1, 'comments': 2, 'fact_table': 3}
        ktypes.sort(key=lambda x: ktype_order[x])
        knowledge = ''
        for ktype in ktypes:
            if ktype == 'fact_table':
                fact_table = data['fact_table']
                ft_str = ''
                if 'box_office' in fact_table:
                    ft_str += ' ' + str(fact_table['box_office'])
                if 'taglines' in fact_table:
                    ft_str += ' ' + list_to_str(fact_table['taglines'])
                if 'awards' in fact_table:
                    ft_str += ' ' + list_to_str(fact_table['awards'])
                if 'similar_movies' in fact_table:
                    ft_str += ' ' + list_to_str(fact_table['similar_movies'])
                knowledge += '\n' + ft_str[1:]
            elif ktype == 'comments':
                knowledge += '\n' + list_to_str(data['comments'])
            else:
                knowledge += '\n' + data[ktype]
    return knowledge[1:]