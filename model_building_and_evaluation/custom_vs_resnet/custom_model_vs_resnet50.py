#Dependencies

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


classes = ["airplabe","automobile","bird","cat","deer","dog","frog","horse","ship","Truck"]

# Load the CIFAR-10 dataset

(train_images,train_labels),(test_images,test_labels)=keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images=train_images/255.0
test_images=test_images/255.0


######################################################################################################################
                                          ### custom model with keras-Tuner ### 
######################################################################################################################

# Define the CNN model

def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(32,32,3)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model

# Initialize tuner and perform hyperparameter tuning
tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output1',project_name="cifar10")

tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)

# Get the best model
model1=tuner_search.get_best_models(num_models=1)[0]

print("model summary ======> ",model1.summary())

history2 = model1.fit(train_images, train_labels, epochs=5, validation_split=0.1, initial_epoch=3)

print("h2 accuracy ====> ",history2.history['accuracy'])

model1.save('custom_model.h5')


######################################################################################################################
                                          ### retrain resnet-50 ### 
######################################################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Flatten

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0

# Load the ResNet-50 model with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
predictions = Dense(10, activation='softmax')(x)

# Combine the base model with the custom classification head
model = keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# history3 = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test))
history3 = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

print("h3 accuracy===============> ", history3.history['accuracy'])


###################################################################################################################
                                # Plots for accuracy and loss comparision #
###################################################################################################################

import matplotlib.pyplot as plt


# Compare the performance with visual graphs
plt.figure(figsize=(6, 24))
plt.subplot(4, 1, 1)
plt.plot(history2.history['accuracy'], label='Model 1 Accuracy')
plt.plot(history3.history['accuracy'], label='Model 2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model 1 vs Model 2 training Accuracy')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(history2.history['val_accuracy'], label='Model 1 Validation Accuracy')
plt.plot(history3.history['val_accuracy'], label='Model 2 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model 1 vs Model 2 validation Accuracy')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(history2.history['loss'], label='Model 1 Loss')
plt.plot(history3.history['loss'], label='Model 2 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 1 vs Model 2 trainning Loss')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(history2.history['val_loss'], label='Model 1 Validation Loss')
plt.plot(history3.history['val_loss'], label='Model 2 Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 1 vs Model 2 validation Loss')
plt.legend()

# Save the plot
plt.savefig('resnet-50_vs_custom_model.png')

model.save('resnet_model.h5')
