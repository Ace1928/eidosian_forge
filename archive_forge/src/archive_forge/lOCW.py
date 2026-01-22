import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-128k-instruct"
model_path = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)


def load_pretrained_model(
    model_path: str, prune: bool = False, quantize: bool = False
) -> tf.keras.Model:
    """
    Loads a pre-trained TensorFlow Keras model, with optional pruning and quantization.

    Args:
        model_path (str): Path to the saved model.
        prune (bool): If True, apply pruning to the model.
        quantize (bool): If True, apply quantization to the model.

    Returns:
        tf.keras.Model: The processed model.
    """
    model = load_model(model_path)
    if prune:
        pruning_params = {
            "pruning_schedule": tf.keras.optimizers.schedules.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=2000,
                end_step=10000,
            )
        }
        model = prune_low_magnitude(model, **pruning_params)
    if quantize:
        model = quantize_model(model)
    return model


def train_model(
    model: tf.keras.Model, dataset: list, epochs: int = 1
) -> tf.keras.Model:
    """
    Trains the model on the provided dataset for a given number of epochs.

    Args:
        model (tf.keras.Model): The model to be trained.
        dataset (list): A list of tuples (input_text, target_text).
        epochs (int): The number of epochs for training.

    Returns:
        tf.keras.Model: The trained model.
    """
    for epoch in range(epochs):
        for input_text, target_text in dataset:
            loss = model.train_on_batch(input_text, target_text)
            print(f"Epoch {epoch}, Loss: {loss}")
    return model


@st.cache(allow_output_mutation=True)
def load_and_prepare_model(model_path: str) -> tuple:
    """
    Loads and prepares a causal language model from Hugging Face Transformers.

    Args:
        model_path (str): Path to the model on local disk.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForCausalLM.from_pretrained(model_path, from_pt=True)
    return model, tokenizer


def preprocess_text(text: str) -> np.ndarray:
    """
    Converts text to an array of ASCII values.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: The processed numpy array.
    """
    return np.array([ord(c) for c in text], dtype=np.int32)


def postprocess_response(predictions: np.ndarray) -> str:
    """
    Converts model predictions to string using ASCII values.

    Args:
        predictions (np.ndarray): The predictions array.

    Returns:
        str: The decoded string.
    """
    return "".join([chr(i) for i in np.argmax(predictions, axis=1)])


# Streamlit app setup
st.title("Adaptive Conversational AI")

user_input = st.text_input("Talk to the AI:")
if user_input:
    input_processed = preprocess_text(user_input)
    input_processed = np.expand_dims(input_processed, 0)
    predictions = model.predict(input_processed)
    response = postprocess_response(predictions)
    st.write(response)
    feedback = st.text_input("Feedback to improve AI:")
    if feedback:
        feedback_processed = preprocess_text(feedback)
        feedback_processed = np.expand_dims(feedback_processed, 0)
        model = train_model(model, [(input_processed, feedback_processed)], epochs=1)

# Save the adapted model
SAVE_PATH = "/home/lloyd/Downloads/local_model_store/Phi-3-adaptive"
model.save_pretrained(SAVE_PATH)

# Load the model
MODEL_PATH = "path_to_pretrained_model.h5"
model = load_pretrained_model(MODEL_PATH, prune=True, quantize=True)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Load model and tokenizer for Hugging Face model
hf_model_path = (
    "/home/lloyd/Downloads/local_model_store/microsoft/Phi-3-mini-128k-instruct"
)
model, tokenizer = load_and_prepare_model(hf_model_path)
