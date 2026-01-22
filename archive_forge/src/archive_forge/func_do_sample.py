import vllm
from vllm.lora.request import LoRARequest
def do_sample(llm, lora_path: str, lora_id: int) -> str:
    prompts = ['Quote: Imagination is', 'Quote: Be yourself;', 'Quote: So many books,']
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None)
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
    return generated_texts