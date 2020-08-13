from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print(train_images.shape)       #this should be (60000, 28, 28)
print(len(train_labels))        #this should be (60000)


#normalize the images now
train_images = (train_images / 255.0) - 0.5
test_images = (test_images / 255.0) - 0.5

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# #reshape images
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

#design model now
model = keras.Sequential([
    keras.layers.Conv2D(3, kernel_size = (5,5), strides= 1, activation = 'relu', padding='valid',
                        input_shape=(28,28,1)),
    keras.layers.MaxPool2D(strides = 2),
    # keras.layers.Dropout(0.3),
    keras.layers.Conv2D(3, kernel_size= (3,3), strides = 1, padding = 'same', activation= 'relu'),
    keras.layers.MaxPool2D(strides= 2),
    # keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation= 'relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#traing the model
graphing = model.fit(train_images, train_labels, validation_split=0.33, epochs=15)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


#make the predictions
predictions= model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]
test_images = np.squeeze(test_images, axis=3)

#graphing loss vs epoch

print(graphing.history.keys())
# summarize graphing for accuracy
plt.plot(graphing.history['acc'])
plt.plot(graphing.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize graphing for loss
plt.plot(graphing.history['loss'])
# plt.plot(graphing.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

model.summary()
model.save('Keras_model.h5')