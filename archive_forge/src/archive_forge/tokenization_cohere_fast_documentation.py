import pickle
from typing import Dict, List, Literal, Union
from tokenizers import processors
from ...pipelines.conversational import Conversation
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from ...utils.versions import require_version
Create a Command-R grounded generation (aka RAG) prompt.

        Once rendered, the prompt instructs the model to generate a response with citations in, based on supplied documents.

        Conceptually, this works in the same way as `apply_chat_format`, but takes additional `documents`
        and parameter `citation_mode` parameters.

        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys and a list of
        documents for the model to ground its response on into a prompt string, or a list of token ids.
        This method will use the tokenizer's `grounded_generation_template` template specified at the class level.
        You can override the default template using the `grounded_generation_template` kwarg but the quality of your results may decrease.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            documents (List[Dict[str, str]): A list of dicts, representing documents or tool outputs to ground your
                generation on. A document is a semistructured dict, wiht a string to string mapping. Common fields are
                `url`, `title`, `snippet` etc but should be descriptive of the key. They will get rendered into the prompt.
            citation_mode: either "accurate" (prompt the model to generate an answer first, then rewrite it with citation
                spans in) or "fast", where the prompt instructs the model to generate an answer with citations in directly.
                The former has higher quality citations, the latter requires fewer tokens to be generated.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

        Returns:
            `str`: A rendered prompt string.
            or if tokenize=True:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.

        Examples:

        ```python
        >> tokenizer = CohereTokenizerFast.from_pretrained('CohereForAI/c4ai-command-r-v01')

        >> # define documents:
        >> documents = [
            { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
            { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
        ]
        >> # define a conversation:
        >> conversation = [
            {"role": "user", "content": "Whats the biggest penguin in the world?"}
        ]
        >> # render the prompt, ready for user to inspect, or for input into the model:
        >> grounded_generation_prompt = tokenizer.apply_grounded_generation_template(conversation, documents=documents, tokenize=False, add_generation_prompt=True)
        >> print(grounded_generation_prompt)
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
        The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

        ## Basic Rules
        You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

        # User Preamble
        ## Task and Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>
        Document: 0
        title: Tall penguins
        text: Emperor penguins are the tallest.

        Document: 1
        title: Penguin habitats
        text: Emperor penguins only live in Antarctica.
        </results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
        Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
        Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
        Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
        Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'''
        ```
        >> inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        >> outputs = model.generate(inputs, max_new_tokens=128)
        >> print(tokenizer.decode(outputs[0]))
        Relevant Documents: 0,1
        Cited Documents: 0,1
        Answer: The Emperor Penguin is the tallest or biggest penguin in the world. It is a bird that lives only in Antarctica and grows to a height of around 122 centimetres.
        Grounded answer: The <co: 0>Emperor Penguin</co: 0> is the <co: 0>tallest</co: 0> or biggest penguin in the world. It is a bird that <co: 1>lives only in Antarctica</co: 1> and <co: 0>grows to a height of around 122 centimetres.</co: 0>
        