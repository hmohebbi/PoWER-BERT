import os
import keras
import codecs
import csv
import numpy as np
from keras_bert import Tokenizer
import unicodedata
import six


class data_parser:

        def __init__(self, 
                     VOCAB_PATH=None,
                     TASK=None,
                     SEQ_LEN=None,
                     DATA_DIR=None,
                     CASED=False):
                
                self.TASK = TASK
                self.SEQ_LEN = SEQ_LEN
                self.DATA_DIR = DATA_DIR

                self.token_dict = {}
                with codecs.open(VOCAB_PATH, 'r', 'utf8') as reader:
                    for line in reader:
                        token = line.strip()
                        self.token_dict[token] = len(self.token_dict)


                self.tokenizer = Tokenizer(self.token_dict, cased=CASED)


        def _read_csv(self, input_file, quotechar=None):

            """Reads a tab separated value file."""
            with open(input_file) as f:
                reader = csv.reader(f, delimiter=",")
                lines = []
                for idx, row in enumerate(reader):
                    label, headline, body = row
                    body = body.replace('\\', ' ')
                    lines.append((body, label))
                return lines


        def convert_to_unicode(self, text):

          """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
          if six.PY3:
            if isinstance(text, str):
              return text
            elif isinstance(text, bytes):
              return text.decode("utf-8", "ignore")
            else:
              raise ValueError("Unsupported string type: %s" % (type(text)))
          elif six.PY2:
            if isinstance(text, str):
              return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
              return text
            else:
              raise ValueError("Unsupported string type: %s" % (type(text)))
          else:
            raise ValueError("Not running on Python2 or Python 3?")


        def encode(self, first, second=None, max_len=None):

                first_tokens = self.tokenizer._tokenize(first)
                second_tokens = self.tokenizer._tokenize(second) if second is not None else None
                self.tokenizer._truncate(first_tokens, second_tokens, max_len)
                tokens, first_len, second_len = self.tokenizer._pack(first_tokens, second_tokens)
                token_ids = self.tokenizer._convert_tokens_to_ids(tokens)
                segment_ids = [0] * first_len + [1] * second_len
                token_len = first_len + second_len
                pad_len = 0
                if max_len is not None:
                    pad_len = max_len - first_len - second_len
                    token_ids += [self.tokenizer._pad_index] * pad_len
                    segment_ids += [0] * pad_len
                input_mask = [1]*token_len+[0]*pad_len
                return token_ids, segment_ids, input_mask


        def get_train_data(self):

                data_path = os.path.join(self.DATA_DIR, "train.csv")
                train_x, train_y = self.load_data(data_path)
                return train_x, train_y


        def get_test_data(self):

                data_path = os.path.join(self.DATA_DIR, "test.csv")
                test_x, test_y = self.load_data(data_path)
                return test_x, test_y


        def load_data(self, data_path):
                
                if self.TASK == 'ag_news':
                        data_x, data_y = self.load_data_agnews(data_path)
                        
                else:
                        raise ValueError('No data loader for the given TASK.')

                return data_x, data_y



        def load_data_agnews(self, path):

                indices, sentiments, masks = [], [], []
                lines = self._read_csv(path)
                for (i, line) in enumerate(lines):
                        text_a = self.convert_to_unicode(line[0])
                        label = self.convert_to_unicode(line[1])
                        ids, segments, mask = self.encode(text_a, max_len=self.SEQ_LEN)
                        indices.append(ids)
                        sentiments.append(label)
                        masks.append(mask)
                items = list(zip(indices, masks, sentiments))
                np.random.shuffle(items)
                indices, masks, sentiments = zip(*items)
                indices = np.array(indices)
                masks = np.array(masks)
                return [indices, np.zeros_like(indices), masks], np.array(sentiments)

