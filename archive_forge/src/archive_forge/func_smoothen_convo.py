from .build import build
import os
def smoothen_convo(conversation, opt):
    """
    Aggregates contiguous responses by the same speaker in the data so that data
    eventually contains alternations between USER and ASSISTANT.

    :param conversation:
        The dialogue between USER and ASSISTANT with possible multiple contiguous
        speeches by the same speaker
    :param opt:
        options dict, mainly useful for accessing value of exclude_invalid_data
    """
    dialogue = conversation['utterances']
    conversation_stack = []
    for speech in dialogue:
        if conversation_stack and speech['speaker'] == conversation_stack[-1]['speaker']:
            conversation_stack.append(join_speech(conversation_stack.pop(), speech))
        else:
            conversation_stack.append(speech)
    processed_conversation = []
    corrupt = False
    for speech in conversation_stack:
        if opt.get('exclude_invalid_data', True) and 'ctr' in speech and (speech['ctr'] > 5):
            corrupt = True
        processed_conversation += [speech]
    return (update_indexes(processed_conversation), corrupt)