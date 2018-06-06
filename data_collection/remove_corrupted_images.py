import os
from PIL import Image

img_dir = r"downloads"

total_images = 0
total_removed = 0
for filename in os.listdir(img_dir):

    filepath = os.path.join(img_dir, filename)
    print('file folder >>>>>'+filepath)
    if filename == ".DS_Store":
        print('DS folder -----'+filepath)
        os.remove(filepath)
    else:
        countF = 0
        countT = 0
        for imagename in os.listdir(filepath):
            imagepath = os.path.join(filepath, imagename)
            total_images += 1

            try:
                with Image.open(imagepath) as im:
                    #print('ok')
                    countT += 1
            except:
                print(imagepath)
                os.remove(imagepath)
                countF += 1
                total_removed += 1

    print("checked in path: ", filepath, " OK Images: ", countT, "Removed: ", countF)
print("ok Total Images: ", total_images)
print("removed Total Images: ", total_removed)
