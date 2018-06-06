from google_images_download import google_images_download


response = google_images_download.googleimagesdownload()   #class instantiation

keywords_list = "bathing bikini, " \
                "bathing suits, " \
                "beachbikini," \
                "beachbikini hot," \
                "beach bra," \
                "beautiful bikini," \
                "bikini," \
                "bikinibeach," \
                "bikiniexotic," \
                "bikinihot," \
                "bikinis," \
                "bikinisexy," \
                "bikini beach," \
                "bikini beach hot," \
                "bikini beach super," \
                "bikini celebrities," \
                "bikini collection," \
                "bikini hot," \
                "bikini mini," \
                "bikini sexy," \
                "bikini swim," \
                "brazilian bikini," \
                "bra bikini," \
                "bra sexy," \
                "cheeky bikini," \
                "exoticbikini," \
                "exotic bikini," \
                "famous bikini," \
                "full bikini," \
                "girls beach," \
                "hotbikini," \
                "hotbra," \
                "hot bra," \
                "instagram bikini," \
                "instagram bikini sexy," \
                "instagram bikini super hot," \
                "instagram hot bikini," \
                "micro bikini," \
                "model bikini," \
                "photo bikini," \
                "pinterest bikini," \
                "pose bikini," \
                "sexybikini," \
                "sexy bikini," \
                "string bikini," \
                "super bikini," \
                "super hot bikini," \
                "super hot bra," \
                "swimwear bikini," \
                "top bikini," \
                "top model bikini," \
                "woman bikini," \
                "woman bikini beach hot," \
                "women bikini," \
                "women swimming suit"


arguments = {"keywords": keywords_list,
                 
             "limit": 20000, "print_urls": True, 'chromedriver': './google_images_download/chromedriver.exe'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images