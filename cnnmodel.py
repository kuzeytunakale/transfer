import tensorflow as tf
import matplotlib.pyplot as plt
import winsound
import datetime
import os
import time
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

# GPU Kontrolü
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"{len(physical_devices)} GPU(lar) bulundu: {[device.name for device in physical_devices]}")    
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU bulunamadı, CPU kullanılacak!")

# Transfer Learning: Önceden Eğitilmiş MobileNet Kullanımı
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = True
for layer in base_model.layers[:100]:  # İlk 100 katmanı dondur
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
output = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


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

checkpoint_best = ModelCheckpoint(
    'records/best_model.keras',    # en iyi modeli buraya kaydeder
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

train_ge_len = len(train_generator)
validation_ge_len = len(validation_generator)

# # Log dosyasının saklanacağı klasörü oluştur
# log_dirr = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")                                 tensorboard_callback
# print(f"Creating directory: {log_dirr}")
# os.makedirs(log_dirr)

# time.sleep(2)
# tensorboard_callback = TensorBoard(log_dir=log_dirr, histogram_freq=1)




# tensorboard = TensorBoard(log_dir="/tmp/tensorboard/{}".format(time), histogram_freq=1)



# # Modelin Eğitilmesi

cnn_history = model.fit(
    train_generator,
    steps_per_epoch=train_ge_len,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_ge_len,
    callbacks=[lr_reduction, early_stopping, checkpoint_best]
)

os.makedirs('records', exist_ok=True)


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

# tensorboard --logdir=logs/fit
