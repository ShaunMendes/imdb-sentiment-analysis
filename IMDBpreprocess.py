import sys
sys.path.append("/home/fractaluser/fastai/courses/ml1")
import torch, cv2
from fastai.nlp import *
PATH='data/aclImdb/'
names = ['neg','pos']
X_train, y_train = texts_labels_from_folders(f'{PATH}train',names)
X_val, y_val = texts_labels_from_folders(f'{PATH}test',names)
train = {'Reviews' : X_train, 'Sentiment': y_train}
val = {'Reviews' : X_val, 'Sentiment': y_val}
train = pd.DataFrame(train)
val = pd.DataFrame(val)
train.to_csv('IMDB_train.csv', index = False)
val.to_csv('IMDB_val.csv', index = False)
