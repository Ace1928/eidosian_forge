import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("model.h5")

# Save the model in the TensorFlow SavedModel format
model.save("model", save_format="tf")

# Start the TensorFlow Serving server
# !tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path=/path/to/model
