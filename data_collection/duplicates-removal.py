import sys
import hashlib
import os


def find_duplicates(folders):
    """
    Takes in an iterable of folders and prints & returns the duplicate files
    """
    dup_size = {}
    for i in folders:
        # Iterate the folders given
        if os.path.exists(i):
            # Find the duplicated files and append them to dup_size
            join_dicts(dup_size, find_duplicate_size(i))
        else:
            print('%s is not a valid path, please verify' % i)
            return {}

    print('Comparing files with the same size...')
    dups = {}
    for dup_list in dup_size.values():
        if len(dup_list) > 1:
            join_dicts(dups, find_duplicate_hash(dup_list))
    print_results(dups)
    return dups


def find_duplicate_size(parent_dir):
    # Dups in format {hash:[names]}
    dups = {}
    for dirName, subdirs, fileList in os.walk(parent_dir):
        #print('Scanning %s...' % dirName)
        print(dirName)
        continue
        for filename in fileList:
            # Get the path to the file
            path = os.path.join(dirName, filename)
            # Check to make sure the path is valid.
            if not os.path.exists(path):
                continue
            # Calculate sizes
            file_size = os.path.getsize(path)
            # Add or append the file path
            if file_size in dups:
                dups[file_size].append(path)
            else:
                dups[file_size] = [path]
    exit(0)
    return dups


def find_duplicate_hash(file_list):
    print('Comparing: ')
    for filename in file_list:
        print('    {}'.format(filename))
    dups = {}
    for path in file_list:
        file_hash = hashfile(path)
        if file_hash in dups:
            dups[file_hash].append(path)
        else:
            dups[file_hash] = [path]
    return dups


# Joins two dictionaries
def join_dicts(dict1, dict2):
    for key in dict2.keys():
        if key in dict1:
            dict1[key] = dict1[key] + dict2[key]
        else:
            dict1[key] = dict2[key]


def hashfile(path, blocksize=65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def print_results(dict1):
    results = list(filter(lambda x: len(x) > 1, dict1.values()))
    if len(results) > 0:
        print('Duplicates Found:')
        print(
            'The following files are identical. The name could differ, but the'
            ' content is identical'
            )
        print('___________________')
        count = 0
        for result in results:
            index = 0
            for subresult in result:
                print('\t\t%s' % subresult)
                index += 1
                if index >= 2:
                    count += 1
                    os.remove(subresult)
            print('___________________')
        print("Duplicates Found:", count)

    else:
        print('No duplicate files found.')


def main():
    find_duplicates(['images'])


if __name__ == '__main__':
    sys.exit(main())