#!/usr/bin/env python
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from numpy import argmax
import h5py
import random

images_path = "less_images"
test_images_path = "test_images"
results_path = "results"


features_path = "load_stuff/features.h5"
labels_path	= "load_stuff/labels.h5"
model_path = "load_stuff/model.h5"


img_width = 128
img_height = 128

num_channel = 3
num_epoch = 20

# Define the number of classes
num_classes = 2
names = ["not_porn", "porn"]

calcMetrics = False

addBlurEffect_Porn = True
addSaltAndPepper = False

save_images = True
im_show_images = False
show_distances = False

test_count_images = 0

if addBlurEffect_Porn:
    results_path += "/blurred/"
else:
    results_path += "/normal/"

if addSaltAndPepper:
    results_path += "salt_n_pepper/"


if save_images:
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print("made dir: ", results_path)



def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



def load_data(path, add_noise=False):
    # Define data path
    imagePaths = sorted(list(paths.list_images(path)))

    img_data_list = []
    labels = []

    count = 0
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)

        input_img_resize = cv2.resize(image, (img_width, img_height))
        if add_noise and addSaltAndPepper:
            input_img_resize = sp_noise(input_img_resize, 0.01)

        img_data_list.append(input_img_resize)

        # extract the class label from the image path and update the  labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "porn" else 0
        labels.append(label)

        if count % 1000 == 0:
            print("[INFO] processed - " + str(count))
            print('Loaded the images of dataset - ' + '{}\n'.format(imagePath))
        count += 1
    return img_data_list, labels


if (not os.path.exists(features_path)) or (not os.path.exists(labels_path)):

    img_data_list, labels = load_data(images_path)

    # save features and labels
    h5f_data = h5py.File(features_path, 'w')
    h5f_data.create_dataset('dataset', data=np.array(img_data_list))

    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset('dataset', data=np.array(labels))

else:
    # import features and labels
    h5f_data = h5py.File(features_path, 'r')
    h5f_label = h5py.File(labels_path, 'r')

    features_string = h5f_data['dataset']
    labels_string = h5f_label['dataset']

    img_data_list = np.array(features_string)
    labels = np.array(labels_string)

    h5f_data.close()
    h5f_label.close()

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255  # normalize from 0 - 1

labels = np.array(labels)
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print("Train length: ", len(X_train), " Test length: ", len(X_test))

input_shape = img_data[0].shape
print("input shape", input_shape)

if not os.path.isfile(model_path):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=[])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])


    print("Training...")
    model.fit(X_train, y_train, batch_size=128, epochs=num_epoch, validation_data=(X_test, y_test), shuffle=True, verbose=2)

    model.save(model_path)
    print("[STATUS] saved model and weights to disk: ", model_path)
else:
    model = load_model(model_path)
    print("[STATUS] model loaded from disk: ", model_path)

# Viewing model_configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

print("Test accuracy = %3.3f" % (model.evaluate(X_test, y_test)[1]))


##################################################
# Get Dense_1 layer from Model

representation_model = keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
feat_train = representation_model.predict(X_train)
feat_test = representation_model.predict(X_test)


##################################################
# Evaluation METRICS

def calculateMetrics(feat_train, y_train_eval, feat_test, y_test_eval):

    nn_eval = NearestNeighbors(n_neighbors=len(y_train_eval), algorithm='brute', metric='cosine')
    nn_eval.fit(feat_train)

    distances_eval, indices = nn_eval.kneighbors(feat_test)

    targets_train = np.cast[np.int](y_train_eval)
    targets_test = np.cast[np.int](y_test_eval)

    relevant_vectors = np.zeros_like(indices)
    for i in range(feat_test.shape[0]):
        relevant_vectors[i, :] = targets_train[indices[i, :]] == targets_test[i]

    N = np.float64(len(targets_test))
    precision = np.cumsum(relevant_vectors, axis=1) / np.arange(1, relevant_vectors.shape[1] + 1)
    precision = np.sum(np.float64(precision), axis=0) / N

    bins = np.bincount(y_train_eval)
    idx = np.nonzero(bins)[0]
    instances_per_target = dict(zip(idx, bins[idx]))

    instances_per_query = np.zeros((y_test_eval.shape[0], 1))
    for i in range(y_test_eval.shape[0]):
        instances_per_query[i] = instances_per_target[y_test_eval[i]]
    recall = np.cumsum(relevant_vectors, axis=1) / instances_per_query
    recall = np.sum(np.float64(recall), axis=0) / N

    return precision, recall



if calcMetrics:

    layers = ['flatten_1', 'dense_1', 'dense_2']

    for layer_name in layers:
        print("caclulating metrics for : ", layer_name)
        representation_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        feat_train = representation_model.predict(X_train)
        feat_test = representation_model.predict(X_test)

        y_train_eval = argmax(y_train, axis=1)
        y_test_eval = argmax(y_test, axis=1)

        precision, recall = calculateMetrics(feat_train, y_train_eval, feat_test, y_test_eval)
        plt.figure(2)
        plt.plot(recall, precision)

    plt.figure(2)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall')
    plt.legend(layers)
    plt.show()



##################################################
# Approximate nearest neighbors

def find_nearest_neighbors(feat):
    find_distances, find_idx, = nn.kneighbors([feat])
    find_idx = find_idx[0]
    return find_distances, find_idx





def TestImage(index, idx_local, distances_local, class_number, x_test_local):
    plt.figure(figsize=(2, 2))
    global test_count_images
    if class_number == 1 and addBlurEffect_Porn:
        image_l = cv2.blur(x_test_local[index], (12, 12))
    else:
        image_l = x_test_local[index]
    plt.imshow(image_l, interpolation="bilinear")




    # classify the input image
    image_label = np.expand_dims(image_l, axis=0)
    (notPorn, porn) = model.predict(image_label)[0]

    # build the label

    label = "Porn" if porn > notPorn else "Not Porn"
    proba = porn if porn > notPorn else notPorn
    #print("{}: {:.2f}%".format(label, proba * 100))
    plt.axis('off')

    class_image_name = names[class_number]
    label = "{:.2f}%".format(proba * 100)
    title = class_image_name + " " + label
    print(title)
    plt.title(title)

    if show_distances:
        print("distances", class_image_name)
        print(distances_local)

    if im_show_images:
        plt.show()
    if save_images:
        plt.savefig(results_path + 'input_test_' + str(test_count_images) + '.png')
        test_count_images += 1

    plt.figure(figsize=(12, 4))
    wrong_assumptions = 0
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        inverted_number = argmax(y_train[idx_local[i]])
        if inverted_number == 1 and addBlurEffect_Porn:
            image_l = cv2.blur(X_train[idx_local[i]], (12, 12))
        else:
            image_l = X_train[idx_local[i]]
        plt.imshow(image_l, interpolation="bilinear")



        if inverted_number != class_number:
            wrong_assumptions += 1

        plt.axis('off')
        plt.title(names[inverted_number])
    print("image: ", test_count_images, " WRONG ASSUMPTIONS:", wrong_assumptions)
    if im_show_images:
        plt.show()
    if save_images:
        plt.savefig(results_path + 'input_test_' + str(test_count_images) + '.png')
        test_count_images += 1




nn = NearestNeighbors(n_neighbors=30, algorithm='brute', metric='cosine')
nn.fit(feat_train)


##################################################
#Show 2 examples: one PORN and one NOT_PORN from X_test


i = 0
showNotPornCount = True
indexNotPornImage = 0

showPornCount = True
indexPornImage = 0

for eachTestFeat in feat_test:
    inverted = argmax(y_test[i])

    if inverted == 0 and showNotPornCount:  # Not porn

        distances_NotPorn, idx_notPorn = find_nearest_neighbors(feat_test[i])

        indexNotPornImage = i
        showNotPornCount = False

    if inverted == 1 and showPornCount:  # Porn

        distances_Porn, idx_Porn = find_nearest_neighbors(feat_test[i])

        indexPornImage = i
        showPornCount = False

    i += 1


##################################################
#Show NOT PORN
if not addSaltAndPepper:

    print(test_count_images)
    TestImage(indexNotPornImage, idx_notPorn, distances_NotPorn, 0, X_test)

    #################################################
    #Show PORN
    TestImage(indexPornImage, idx_Porn, distances_Porn, 1, X_test)
else:
    test_count_images += 4




#################################################
#TEST FROM FOLDER
#################################################

img_data_list_test, labels_test = load_data(test_images_path, add_noise=True)


img_data_test = np.array(img_data_list_test)
img_data_test = img_data_test.astype('float32')
img_data_test /= 255  # normalize from 0 - 1


labels = np.array(labels_test)
# convert class labels to on-hot encoding
Y_test = np_utils.to_categorical(labels, num_classes)


#Shuffle the dataset
x_test, y_test = shuffle(img_data_test, Y_test, random_state=2)

feats_test = representation_model.predict(x_test)

count = 0
for eachImage in x_test:
    distances, idx_n = find_nearest_neighbors(feats_test[count])
    class_image = argmax(y_test[count])
    TestImage(count, idx_n, distances, class_image, x_test)
    count += 1

