import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter


model = tf.keras.models.load_model("records\\model.keras")

# i = 1
# c= []
# d = []

# while i < 350: #Burada test görselinin bulunduğu klasördeki öğelerin sayısı + 1 olmalı

#     img_path = 'dataset\\test\\2-plastik\\' + str(i) + '.jpg'  # "3-green-glass" yerine istenilen klasör yazılmalı
#     img = load_img(img_path, target_size=(224, 224))

#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)


#     prediction = model.predict(img_array)


#     c.append(f"Predicted class: {np.argmax(prediction)} with probabilities: {prediction}")
#     d.append(np.argmax(prediction))
#     i = i + 1
#     print(i)

# for a in c:
#     print(a)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/test/',
    labels=None,               # zaten tahmin yapacağız
    image_size=(224, 224),
    batch_size=64,             # bir kerede kaç resim
    shuffle=False              # sıralamayı koru
)

predictions = model.predict(test_ds)  # her batch’i sırayla alır

# 4. Sonuçları düzleştir ve sınıfları al
predicted_classes = predictions.argmax(axis=1)

# 5. Sınıf dağılımı
counts = Counter(predicted_classes)
for cls, cnt in counts.items():
    print(f"Class {cls}: {cnt} adet")




# sayi_sayaci = Counter(d)
# for sayi, adet in sayi_sayaci.items():
#     print(f"{sayi}: {adet} kez")
