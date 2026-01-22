from mlc_chat import ChatModule
import logging
def setup_mlc_llm_optimized():
    try:
        cm = ChatModule(model='optimized_model_path', model_lib_path='optimized_model_lib_path')
        return cm
    except Exception as e:
        logging.error(f'Error setting up optimized MLC-LLM: {e}')
        return None