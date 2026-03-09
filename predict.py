import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("animal_disease_model.h5")

classes = ["Healthy","Lumpy Skin Disease","Mastitis"]

img = image.load_img("test.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array/255.0

result = model.predict(img_array)

prediction = classes[np.argmax(result)]

print("Predicted Disease:", prediction)