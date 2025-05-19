import tensorflow as tf
import numpy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

pred_list = []

model = tf.keras.models.load_model("./records/best_model.keras")

img_path = "./dataset/test/6-cardboard/11.jpg"
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = numpy.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(f"Predicted class: {numpy.argmax(prediction)} with probabilities:")

for predict in prediction:
    for pred in predict:
        pred_list.append(pred)
        print(pred)





match numpy.argmax(prediction):
    case 0:
       a = "Metal atık"
    case 1:
        a = "Plastik atık"
    case 2:
        a = "Cam atık"
    case 3:
        a = "Pil atık"
    case 4:
        a = "Biyolojik atık"
    case 5:
        a = "Karton atık"
    case 6:
        a = "Kağıt atık"


print(a)
