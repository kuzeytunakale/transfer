import tensorflow as tf
import matplotlib.pyplot as plt
import winsound
import datetime
import os
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

model = tf.keras.models.load_model('records/model.keras')

model.trainable = True
for layer in model.layers[:100]:  # İlk 100 katmanı dondur
    layer.trainable = False
   
# Modelin Derlenmesi
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model Özeti
model.summary()
# Veri Artırma (Data Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Eğitim Çağrıları
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    #verbose=1,
    factor=0.5,
    min_lr=1e-6
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

train_ge_len = len(train_generator)
validation_ge_len = len(validation_generator)

# # Log dosyasının saklanacağı klasörü oluştur
# log_dirr = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                                 tensorboard_callback
# print(f"Creating directory: {log_dirr}")
# os.makedirs(log_dirr)

# time.sleep(2)
# tensorboard_callback = TensorBoard(log_dir=log_dirr, histogram_freq=1)

tensorboard = TensorBoard(log_dir="/tmp/tensorboard/{}".format(time),
                        batch_size=32,
                        histogram_freq=1,
                        write_grads=False)


# # Modelin Eğitilmesi

cnn_history = model.fit(
    train_generator,
    steps_per_epoch=train_ge_len,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_ge_len,
    callbacks=[lr_reduction, early_stopping, tensorboard]
)


model.save('records/model_h5.h5')


model.save('records/model.keras')

print(cnn_history.history)

os.makedirs('recorded_images', exist_ok=True)

# Eğitim Sonuçlarının Görselleştirilmesi
plt.plot(cnn_history.history['loss'], label='Training Loss', color='blue')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('recorded_images/mobilenet_training_validation_loss.png')
plt.close()

plt.plot(cnn_history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('recorded_images/mobilenet_training_validation_accuracy.png')
plt.close()

# Sesli Bildirim
for _ in range(5):
    frequency = 2500  
    duration = 1500  
    winsound.Beep(frequency, duration)
