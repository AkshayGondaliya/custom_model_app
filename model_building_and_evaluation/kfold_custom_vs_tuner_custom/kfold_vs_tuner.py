#Dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# K-fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True)
cvscores = []

for train, val in kfold.split(train_images, train_labels):
    model = create_model()
    model.fit(train_images[train], train_labels[train], epochs=5, batch_size=64, verbose=0)
    scores = model.evaluate(train_images[val], train_labels[val], verbose=0)
    print(f'{model.metrics_names[1]}: {scores[1]*100}')
    cvscores.append(scores[1] * 100)

print('Average Accuracy: ', np.mean(cvscores))

# Train the model on the entire training data
model = create_model()
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy: ", test_acc)

# Confusion Matrix
test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)
cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(cm)

 # Generate classification report
print('Classification Report')
print(classification_report(test_labels, test_predictions))

# Plot training & validation accuracy values
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('kfold_plots.png')

model.save('kfold_custom_model.h5')


###################################################################################################################
                                     # Keras-tuner implementation #
###################################################################################################################

from tensorflow import keras
from keras_tuner.tuners import RandomSearch

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(layers.Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize tuner and perform hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cifar10'
)

tuner.search_space_summary()

tuner.search(train_images, train_labels, epochs=5, validation_split=0.1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

history2 = best_model.fit(train_images, train_labels, epochs=10, validation_split=0.1, initial_epoch=3)

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Confusion matrix
predictions = np.argmax(best_model.predict(test_images), axis=1)
cm = confusion_matrix(test_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Generate classification report
print('Classification Report')
print(classification_report(test_labels, predictions))

# Plot training & validation accuracy values
plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('tuner_plots.png')

best_model.save('tuner_custom_model.h5')

###################################################################################################################
                                # Plots for accuracy and loss comparision #
###################################################################################################################


# Comparison of training & validation accuracy and training & validation loss of both models using graph visualization.
plt.figure(figsize=(6, 24))

plt.subplot(4, 1, 1)
plt.plot(history.history['accuracy'], label='k-fold Accuracy')
plt.plot(history2.history['accuracy'], label='keras-tuner Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('k-fold vs keras-tuner training Accuracy')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(history.history['val_accuracy'], label='k-fold Validation Accuracy')
plt.plot(history2.history['val_accuracy'], label='keras-tuner Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('k-fold vs keras-tuner validation Accuracy')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(history.history['loss'], label='k-fold Loss')
plt.plot(history2.history['loss'], label='keras-tuner Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('k-fold vs keras-tuner trainning Loss')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(history.history['val_loss'], label='k-fold Validation Loss')
plt.plot(history2.history['val_loss'], label='keras-tuner Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('k-fold vs keras-tuner validation Loss')
plt.legend()

# Save the plot
plt.savefig('k_fold_vs_keras_tuner.png')