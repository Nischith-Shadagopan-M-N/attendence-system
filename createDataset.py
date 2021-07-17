import os
import shutil

if not os.path.exists('dataset'):
    os.makedirs('dataset')

peopleNames = os.listdir('./IMFDB_final/')

for personName in peopleNames:
    imageIndex = 0
    if not os.path.exists('dataset/' + personName):
        os.makedirs('dataset/' + personName)
    
    for movieName in os.listdir('./IMFDB_final/' + personName + '/'):
        if os.path.exists('./IMFDB_final/' + personName + '/' + movieName + '/images/'):
            for filename in os.listdir('./IMFDB_final/' + personName + '/' + movieName + '/images/'):
                shutil.copyfile('./IMFDB_final/' + personName + '/' + movieName + '/images/' + filename, 
                                './dataset/' + personName + "/" + personName + '_' + str(imageIndex) + '.jpg')
                imageIndex += 1