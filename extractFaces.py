from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

detector = MTCNN()

def extract(filename, required_size=(160, 160)):
    pixels = np.asarray(Image.open(filename))
    results = detector.detect_faces(pixels)
    print(results)

    faces = []
    
    for face in results:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        image = Image.fromarray(pixels[y1:y2, x1:x2])
        image = image.resize(required_size)
        faces.append(image)
    return faces

index = 0
for filename in os.listdir('./pictures/'):
    faces = extract('./pictures/' + filename)
    for face in faces:
        face.save('./cropped/' + str(index) + '.jpg')
        index += 1
    






"""

classroom = Image.open('classroom.jpeg')
drawing = ImageDraw.Draw(classroom)
results = extract(classroom)
names = ["Nischit Shadagopan", "Abhinav Hampiholi", "Vishnu Kiran", "Anirudh Ajith"]
index = 0
for result in results:
    x1, y1, l, w = result['box']
    x2, y2 = x1 + l, y1 + w
    drawing.rectangle(((x1,y1),(x2,y2)), fill=None, width=2, outline='yellow')
    drawing.text( (x1,y1-15), names[index])
    index += 1

classroom.show()
classroom.save('classroom2.jpeg')

"""