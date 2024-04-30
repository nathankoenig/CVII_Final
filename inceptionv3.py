# Import the libraries
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers

# Get the pretrained model
pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

# Freeze and unfreeze layers for pre-training (311 total)
for layer in pre_trained_model.layers[:180]:
    layer.trainable = False
for layer in pre_trained_model.layers[180:]:
    layer.trainable = True

# Add layers for training
x = layers.Flatten()(pre_trained_model.output)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4, activation='softmax')(x)
model = Model(inputs=pre_trained_model.input, outputs=x)

# Create learning rate schedule
model.compile(optimizer=Adam(learning_rate=1e-4, weight_decay=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generator
train_datagen = ImageDataGenerator(rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory('train', batch_size=20, class_mode='categorical', target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory('validation', batch_size=20, class_mode='categorical', target_size=(150, 150))

# Added training params
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

# Training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=150,
    validation_steps=50,
    verbose=2,
    callbacks=callbacks
)

# Training and validation accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Training and validation loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Save the created plots
plt.savefig('model_performance.png')
