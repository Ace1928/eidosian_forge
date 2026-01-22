import copy
import datasets
def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset('samsum', split=split)
    prompt = f'Summarize this dialog:\n{{dialog}}\n---\nSummary:\n'

    def apply_prompt_template(sample):
        return {'prompt': prompt.format(dialog=sample['dialogue']), 'summary': sample['summary']}
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample['prompt'], add_special_tokens=False)
        summary = tokenizer.encode(sample['summary'] + tokenizer.eos_token, add_special_tokens=False)
        sample = {'input_ids': prompt + summary, 'attention_mask': [1] * (len(prompt) + len(summary)), 'labels': [-100] * len(prompt) + summary}
        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset