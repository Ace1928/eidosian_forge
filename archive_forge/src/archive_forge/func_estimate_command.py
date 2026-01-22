from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def estimate_command(args):
    data = gather_data(args)
    for row in data:
        for i, item in enumerate(row):
            if isinstance(item, (int, float)):
                row[i] = convert_bytes(item)
    headers = ['dtype', 'Largest Layer', 'Total Size', 'Training using Adam']
    title = f'Memory Usage for loading `{args.model_name}`'
    table = create_ascii_table(headers, data, title)
    print(table)