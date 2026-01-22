import os
from parlai.core.build_data import modelzoo_path
def override_args(opt, override_opt):
    args = set(['embedding_file', 'embedding_dim', 'hidden_size', 'doc_layers', 'question_layers', 'rnn_type', 'optimizer', 'concat_rnn_layers', 'question_merge', 'use_qemb', 'use_in_question', 'use_tf', 'vocab_size', 'num_features', 'use_time'])
    for k, v in override_opt.items():
        if k in args:
            opt[k] = v