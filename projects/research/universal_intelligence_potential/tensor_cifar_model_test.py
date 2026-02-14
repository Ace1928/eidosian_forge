import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging
import time
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    logging.info("Loading and preprocessing CIFAR-10 dataset")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def create_data_augmentation():
    logging.info("Creating data augmentation generator")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    return datagen

def build_model(input_shape):
    logging.info("Building the model")
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, datagen, train_images, train_labels, test_images, test_labels):
    logging.info("Training the model")
    datagen.fit(train_images)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),  # Changed .h5 to .keras
        TensorBoard(log_dir='./logs')
    ]
    
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                        epochs=50,
                        validation_data=(test_images, test_labels),
                        steps_per_epoch=train_images.shape[0] // 64,
                        callbacks=callbacks)
    return history

def main():
    start_time = time.time()
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    datagen = create_data_augmentation()
    model = build_model(input_shape=(32, 32, 3))
    history = train_model(model, datagen, train_images, train_labels, test_images, test_labels)
    logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
