# Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras import backend as K
import os

filepath='multilabel(new)2.h5'


# Initialising the CNN
classifier = Sequential()
#classifier.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))


# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),  dim_ordering="th"))

classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),  dim_ordering="th"))

classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),  dim_ordering="th"))

# Step 3 - Flattening

#classifier.add(Dropout(0.2, input_shape=(64, 64, 3)))

# Step 4 - Full connection
classifier.add(Flatten(name='flatten'))
classifier.add(Dense(128, input_dim=4, activation='relu'))

#classifier.load_weights(by_name=True,filepath = filepath)
classifier.add(Dense(12, activation = 'sigmoid'))



# Compiling the CNN
classifier.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('Train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Test',
                                            target_size = (128, 128),
                                            batch_size = 32,
 
                                         class_mode = 'categorical')

#classifier.summary()

'''
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

callback = ModelCheckpoint(filepath,
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                mode='auto')
'''
classifier.fit_generator(training_set,
                         samples_per_epoch = 14879,
                         #callbacks=[callback], 
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 4642)
    

# saving model using to step method

classifier.summary()

#wts=classifier.load_weights('multilabel_wts.hdf5')
classifier.save(filepath);
#print(wts);
#using two step method



'''
from keras.models import Model,load_model

#os.chdir('D:/data_science/cnn_feature')

model = load_model('melanoma.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('flatten').output)
intermediate_layer_model.summary()

import h5py
def read_hdf5():

    weights = {}

    keys = []
    with h5py.File('melanoma.h5', 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights

wts=read_hdf5();
print(wts)
'''