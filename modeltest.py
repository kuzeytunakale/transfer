import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pygame
import time

videoVievend= 0
predlist = []
possibility = 0

# Modeli yukleme
model = tf.keras.models.load_model("./records/best_model.keras")

# Tensorflow hızlı başlangıcı
@tf.function
def infer(x):
    return model(x, training=False)

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = infer(dummy)

# Pygame hızlı başlangıcı
pygame.init()

print("Sistem başlıyor...")

while True:

    predlist.clear()

    # kamera sistemi
    camera = cv2.VideoCapture(0)

    while (True):
        ret, videoViev = camera.read()
        cv2.imshow("Computer_Camera", videoViev)

        if cv2.waitKey(50) & 0xFF == 32:
            videoVievend = videoViev
            cv2.imwrite("./recorded_images/camera's_picrure.jpg", videoViev)
            break

    camera.release()
    cv2.destroyAllWindows()

    # Görüntü işleme ve tahmin
    img_path = "./recorded_images/camera's_picrure.jpg"
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # prediction = model.predict(img_array)

    prediction = infer(img_array)

    print(f"Predicted class: {np.argmax(prediction)} with probabilities:")

    for predict in prediction:
        for pred in predict:
            predlist.append(pred)
            print(f"{pred * 100:.20f}") 

    possibility = max(predlist) * 100

    predictionEnd = np.argmax(prediction)

    if predictionEnd == 0:
        conculusion = "Metal atık"
    elif predictionEnd == 1:
        conculusion = "Plastik atık"
    elif predictionEnd == 2:
        conculusion = "Cam atık"
    elif predictionEnd == 3:
        conculusion = "Pil atığı"
    elif predictionEnd == 4:
        conculusion = "Karton atık"
    else:
        conculusion = "Kağıt atık"

    print(conculusion)

    screen = pygame.display.set_mode((1200, 500))
    clock = pygame.time.Clock()
    img = pygame.image.load("./recorded_images/camera's_picrure.jpg")
    img = pygame.transform.scale(img, (550, 400))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.display.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Boşluk tuşuna basılırsa çık
                    running = False
                    pygame.display.quit()

        if running == True: 
            screen.fill((100,100,100))

            font = pygame.font.Font(None, 45)
            font2 = pygame.font.Font(None, 30)

            text = font.render(f"Bu atık %{possibility:.2f} ihtimalle bir {conculusion}", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(300, 100)))
            
            text = font2.render("İhtimaller:", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(300, 200)))

            i = 1
            w = " "

            for pred in predlist:
                
                if i == 1:
                    w = "Metal atık"
                if i == 2:
                    w = "Plastik atık"
                if i == 3:
                    w = "Cam atık"
                if i == 4:
                    w = "Pil atığı"
                if i == 5:
                    w = "Karton atık"
                if i == 6:
                    w = "Kağıt atık"

                text = font2.render(f"{w} ihtimali => %{pred * 100}", True, (255, 255, 255))
                screen.blit(text, text.get_rect(center=(300, (200 + i * 30))))

                i = i + 1

            rect = img.get_rect()
            rect.center = 900, 250
            screen.blit(img, rect)

            pygame.display.flip()
            clock.tick(60)



pygame.quit()