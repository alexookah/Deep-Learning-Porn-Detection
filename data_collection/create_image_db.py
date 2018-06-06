import os
import cv2
import csv
import numpy as np
from glob import glob

img_dir = r"downloads"
class_name = 'bikini'
img_dir_downscaled = "downloads_downscaled"

count = 0


if not os.path.exists(img_dir_downscaled):
    os.makedirs(img_dir_downscaled)
    print("made dir: ", img_dir_downscaled)


def convertGifsToJPG():

    #Conver GIF's TO JPG
    gifs = glob('./**/*.gif', recursive=True)

    for j in gifs:
        img = cv2.imread(j)
        print("successfully converted gif to jpg: ", j[:-3] + 'jpg')
        cv2.imwrite(j[:-3] + 'jpg', img)
        os.remove(j[:-3] + 'gif')
    exit(0)

#convertGifsToJPG()

def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img



with open('downloaded_images_google.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    folder_count = 0
    for folder in os.listdir(img_dir):

        folder_to_create = folder
        if " " in folder_to_create:
            folder_to_create = folder_to_create.replace(" ", "_")

        if not os.path.exists(img_dir_downscaled + "/" + folder_to_create):
            os.makedirs(img_dir_downscaled + "/" + folder_to_create)
            print("made dir: ", img_dir_downscaled + "/" + folder_to_create)

        for filename in os.listdir(img_dir + "/" + folder):

            filename_to_create = filename
            if " " in filename_to_create:
                filename_to_create = filename_to_create.replace(" ", "_")

            if count == 0:
                filewriter.writerow(['ID', 'Category', 'Filename', 'Class'])
            else:

                imageRead = cv2.imread(img_dir + "/" + folder + "/" + filename)

                print(img_dir + "/" + folder + "/" + filename)

                #remove images if 0 kb
                try:
                    imageRead.shape
                    print("checked for shape".format(imageRead.shape))
                except AttributeError as e:
                    pass

                    print("shape not found")
                    print(imageRead)
                    os.remove(img_dir + "/" + folder + "/" + filename)
                    continue
                    # code to move to next frame
                    
                filewriter.writerow([count, folder, filename_to_create, class_name])
                imageTransformed = resizeAndPad(imageRead, (180, 320), 255)
                cv2.imwrite(img_dir_downscaled + "/" + folder_to_create + "/" + filename_to_create, imageTransformed)


            if count == -100:
                break

            count += 1
