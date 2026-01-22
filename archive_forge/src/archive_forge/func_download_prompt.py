import re
from ..utils import cached_file
def download_prompt(prompt_or_repo_id, agent_name, mode='run'):
    """
    Downloads and caches the prompt from a repo and returns it contents (if necessary)
    """
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO
    if re.search('\\s', prompt_or_repo_id) is not None:
        return prompt_or_repo_id
    prompt_file = cached_file(prompt_or_repo_id, PROMPT_FILES[mode], repo_type='dataset', user_agent={'agent': agent_name})
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()