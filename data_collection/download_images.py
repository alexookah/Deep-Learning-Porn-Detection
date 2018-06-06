import csv
from requests import get  # to make GET request


def download(url, file_name=None):
    if not file_name:
        file_names = url.split('/')[5:]
        file_name = '_'.join([str(x) for x in file_names])
    # open in binary mode
    with open("images/porn" + file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


with open('porn-data/porn_tumbzilla_labels.csv', 'r', encoding="utf8") as f:
    reader = csv.reader(f)
    your_list = list(reader)


index = 0
for eachLine in your_list:
    if index == 0:
        index += 1
        continue

    url = eachLine[1]
    print(url)
    download(url)
    index += 1

    print("download progress: ", index, " out of: ", len(your_list))


