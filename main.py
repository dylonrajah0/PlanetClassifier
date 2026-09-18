import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

# Define paths
base_dir = 'TrainingNoPre'

# Data generators with data augmentation
# Note: We're omitting horizontal_flip and rotation since they're already in your preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,  # Zooming
    width_shift_range=0.2,  # Shifting the image widthwise
    height_shift_range=0.2,  # Shifting the image heightwise
    shear_range=0.2,  # Shearing
    fill_mode='nearest',
    validation_split=0.3  # Splitting data: 70% for training, 30% for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # Image size expected by ResNet50
    batch_size=20,
    class_mode='categorical',
    subset='training'  # Specify as training data
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    subset='validation'  # Specify as validation data
)



        # Load ResNet-50 base model, excluding the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers with Dropout
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add a Dropout layer with a 50% dropout rate
predictions = Dense(8, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the validation set
validation_generator.shuffle = False  # Disable shuffling for evaluation
validation_generator.reset()  # Reset the generator to be sure it starts from the beginning

# Predict the validation dataset
Y_pred = model.predict(validation_generator, steps=len(validation_generator))
Y_pred_classes = np.argmax(Y_pred, axis=1)  # Convert predictions classes to labels

# Get the true labels
Y_true = validation_generator.classes

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print('Classification Report')
print(classification_report(Y_true, Y_pred_classes, target_names=validation_generator.class_indices.keys()))

