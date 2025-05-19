import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("records\\model.keras")

sayac_verisi = []


predict_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = predict_datagen.flow_from_directory(
'dataset/test/1-metal',
target_size=(224, 224),
batch_size=1,
class_mode='categorical'
)

predictions = model.predict(train_generator)

print("------------------------------------")

for prediction in predictions:
    sayac_verisi.append(np.argmax(prediction))

sayac = Counter(sayac_verisi)
for sayi, adet in sayac.items():
    print(f"{sayi}: {adet} kez")