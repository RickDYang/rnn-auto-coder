import numpy
import string
import re
import os

# we use ascii as vocabulary
# so we use ascii code as numeric mapping for linux source code
vocabulary_size = len(string.printable)
char2id_map = dict([(x, string.printable.index(x)) for x in string.printable])

def char2id(char):
    '''convert a charater to an int
    return ascii code if the charactor in vocabulary, 
    otherwise return ascii code of space
    Keyword argument:
        char -- charater to convert
    '''
    if char in char2id_map:
        return char2id_map[char]
    else:
        #replace to space ' ' if not ascii
        return char2id_map[' ']

def id2char(id):
    '''convert int id to a charater
    return converted charater
    Keyword argument:
        id -- int id to convert
    '''
    return string.printable[id]

def str2input(str):
    '''convert string to a m x n matrix
    return vector matrix presents the string
    Keyword argument:
        str -- string to convert
    '''
    input = numpy.zeros((len(str), vocabulary_size))
    for i, char in enumerate(str):
        input[i][char2id(char)] = 1
    return input

class code_data:
    def __init__(self, files, looping = False):
        '''constructor
        Keyword argument:
            files -- files where the source code from
        '''
        self.files = files
        self.looping = looping

        self.words = []
        self.cur_file_idx = 0
        self.cur_content_idx = 0

        self.cache_x = None
        self.cache_y = None

    def next_batch(self, batch_size, steps):
        '''get next batch data for input & output
        return tuple, which 1st is input data which size is m x s x n
        which 2nd is output data which size is m x n
        Keyword argument:
            batch_size -- number of data to get
            steps - number of characters in one input data
        '''
        x = numpy.ndarray((0, steps, vocabulary_size), dtype=int)
        y = numpy.ndarray((0, vocabulary_size), dtype=int)
        for i in range(batch_size):
            word = self.get(steps + 1)
            if word == None:
                return x, y

            x = numpy.append(x, str2input(word[:-1]).reshape(1,steps, vocabulary_size) , axis = 0)
            y = numpy.append(y, str2input(word[-1]).reshape(1, vocabulary_size), axis = 0)

        return x, y

    def get_cache_batch(self, batch_size, steps):
        '''get cached batch data for input & output. 
        Once this funtion called, it will return the same data.
        These cached data is for testing/validation purpose
        return tuple, which 1st is input data which size is m x s x n
        which 2nd is output data which size is m x n
        Keyword argument:
            batch_size -- number of data to get
            steps - number of characters in one input data
        '''
        if self.cache_x is not None and self.cache_y is not None:
            return self.cache_x, self.cache_y

        self.cache_x, self.cache_y = self.next_batch(batch_size, steps)

        return self.cache_x, self.cache_y

    def get(self, length):
        '''get a text by length
        '''
        # no word cached, read a new file
        if len(self.words) == 0:
            # if no more, we start again
            if self.cur_file_idx >= len(self.files):
                # no looping
                if self.looping:
                    self.cur_file_idx = 0
                else:
                    return None

            #print("new source file:", self.source_files[self.cur_file_idx])
            self.words = self.read_file(self.files[self.cur_file_idx])
            self.cur_file_idx += 1
            self.cur_content_idx = 0

        max_len = len(self.words)
        start = self.cur_content_idx
        end = start + length
        if end <= max_len:
            self.cur_content_idx += 1
            word = self.words[start:end]
            if (end == max_len):
                #reset words to read file
                self.words = []
            return word
        else:
            # read to end of file
            # fill to spaces
            word = self.words[start:-1]
            word += [' '] * (length - len(word))
            word += ['\n']

            #reset words to read file
            self.words = []
            return word

    def read_file(self, path):
        words = []
        with open(path) as f:
            text = f.read()
            text = self.comment_remover(text)
            #print(text)
            words += text
        return words

    def comment_remover(self, text):
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return "" # note: a space and not an empty string
            else:
                return s

        # remove c comments
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        text = re.sub(pattern, replacer, text)

        #remove space line
        text = os.linesep.join([s for s in text.splitlines() if not re.match(r'^\s*$', s)])
        return text
