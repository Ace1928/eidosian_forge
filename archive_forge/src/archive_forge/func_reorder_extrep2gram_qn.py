from parlai.core.torch_agent import TorchAgent
from .controls import eval_attr
def reorder_extrep2gram_qn(n_best_beam_preds, history, dictionary, verbose):
    """
    Inputs:
        n_best_beam_preds: list length num_candidates of (prediction, score) pairs.
          prediction is a tensor of word indices, score is a single float tensor.
        history: ConvAI2History
        dictionary: parlai DictionaryAgent
        verbose: bool. If True, print out the selection process.

    Outputs: (tensor, tensor) pair which is the chosen (prediction, score)
    """
    if verbose:
        print('persona: ', history.persona_lines)
        print('partner_utts: ', history.partner_utts)
        print('own_utts: ', history.own_utts)
    candidates = []
    if verbose:
        print('\nORIGINAL ORDER:')
    for idx, (pred, score) in enumerate(n_best_beam_preds):
        text = dictionary.vec2txt(pred.tolist())
        text = text.replace('__start__ ', '').replace(' __end__', '')
        if verbose:
            print('%i  %.4f  %s' % (idx, score, text))
        extrep_2gram = eval_attr(text, history, 'extrep_2gram')
        candidates.append((idx, pred, text, score, extrep_2gram))
    candidates = sorted(candidates, key=lambda x: (x[4], x[0]))
    if verbose:
        print('\nSORTED BY EXTREP_2GRAM:')
        for idx, _, text, _, extrep_2gram in candidates:
            print('%i  %.4f  %s' % (idx, extrep_2gram, text))
        print('')
    for _, pred, text, score, _ in candidates:
        if '?' not in text:
            continue
        return (pred, score)
    _, pred, score, _, _ = candidates[0]
    return (pred, score)