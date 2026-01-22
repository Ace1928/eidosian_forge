from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def create_cd(label=None, cat_features=None, text_features=None, embedding_features=None, weight=None, baseline=None, doc_id=None, group_id=None, subgroup_id=None, timestamp=None, auxiliary_columns=None, feature_names=None, output_path='train.cd'):
    _from_param_to_cd = {'label': 'Label', 'weight': 'Weight', 'baseline': 'Baseline', 'doc_id': 'DocId', 'group_id': 'GroupId', 'subgroup_id': 'SubgroupId', 'timestamp': 'Timestamp'}
    _column_description = defaultdict(lambda: ['Num', ''])
    for key, value in locals().copy().items():
        if not (key.startswith('_') or value is None):
            if key in ('cat_features', 'text_features', 'embedding_features', 'auxiliary_columns'):
                if isinstance(value, int):
                    value = [value]
                for index in value:
                    if not isinstance(index, int):
                        raise CatBoostError('Unsupported index type. Expected int, got {}'.format(type(index)))
                    if index in _column_description:
                        raise CatBoostError('The index {} occurs more than once'.format(index))
                    if key == 'cat_features':
                        _column_description[index] = ['Categ', '']
                    elif key == 'text_features':
                        _column_description[index] = ['Text', '']
                    elif key == 'embedding_features':
                        _column_description[index] = ['NumVector', '']
                    else:
                        _column_description[index] = ['Auxiliary', '']
            elif key not in ('feature_names', 'output_path'):
                if not isinstance(value, int):
                    raise CatBoostError('Unsupported index type. Expected int, got {}'.format(type(value)))
                if value in _column_description:
                    raise CatBoostError('The index {} occurs more than once'.format(value))
                _column_description[value] = [_from_param_to_cd[key], '']
    if feature_names is not None:
        for feature_column_index, name in feature_names.items():
            if _column_description[feature_column_index][0] not in ('Num', 'Categ', 'Text', 'NumVector'):
                raise CatBoostError('feature_names contains index {} that does not correspond to feature column'.format(feature_column_index))
            _column_description[feature_column_index][1] = name
    with open(fspath(output_path), 'w') as f:
        for index, (title, name) in sorted(_column_description.items()):
            f.write('{}\t{}\t{}\n'.format(index, title, name))