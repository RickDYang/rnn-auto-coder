import os
from code_data import *

class linux_code:
    def __init__(self, root, ext=[], looping = True):
        '''constructor
        looping
        Keyword argument:
        root -- root directory to get source code file
        ext -- file extions to filters
        '''
        source_files, totalsize = self.get_source_files(root, ext)
        print('total files: {} total size: {}'.format(len(source_files), totalsize))
        
        #divid into training & validation sets in 90%/10%
        validation_size = len(source_files) // 10;
        self.validation_data = code_data(source_files[:validation_size], looping)
        self.training_data = code_data(source_files[validation_size:], looping)

    def get_source_files(self, root, ext):
        source_files = []
        totalsize = 0
        for path, subdirs, files in os.walk(root):
            for name in files:
                if any(name.endswith(x) for x in ext):
                    f = os.path.join(path, name)
                    source_files.append(f)
                    totalsize += os.path.getsize(f)

        return source_files, totalsize
