from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def gather_data(args):
    """Creates an empty model and gathers the data for the sizes"""
    try:
        model = create_empty_model(args.model_name, library_name=args.library_name, trust_remote_code=args.trust_remote_code)
    except (RuntimeError, OSError) as e:
        library = check_has_model(e)
        if library != 'unknown':
            raise RuntimeError(f'Tried to load `{args.model_name}` with `{library}` but a possible model to load was not found inside the repo.')
        raise e
    total_size, largest_layer = calculate_maximum_sizes(model)
    data = []
    for dtype in args.dtypes:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype == 'float16':
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == 'int8':
            dtype_total_size /= 4
            dtype_largest_layer /= 4
        elif dtype == 'int4':
            dtype_total_size /= 8
            dtype_largest_layer /= 8
        dtype_training_size = dtype_total_size * 4
        data.append([dtype, dtype_largest_layer, dtype_total_size, dtype_training_size])
    return data