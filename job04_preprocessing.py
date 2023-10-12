import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True)     # 유니코드 사용너비 조절
df = pd.read_csv('./crawling_data/naver_news_titles_20231012.csv')
print(df.head())
df.info()

# 라벨링   카테고리 분류 - onehot encoding 하기위한 라벨링
X = df['titles']
Y = df['category']

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])
label = encoder.classes_
print(label)
with open ('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

# one hot encoding
onehot_y = to_categorical(labeled_y)
# print(onehot_y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)       # stem=True -> 원형으로 바꿔준다,
    # 형태소들의 리스트로 쭉 바뀐다.
# print(X)

# 의미없는 단어들 빼는 작업(불용어) - stopword
stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list (stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
# print(X[0])

# 새로운 문장을 tokenizing 하기 위해서도 token이 필요하다.
token = Tokenizer()         # 형태소 라벨 세트를 token이 가지고 있다.
token.fit_on_texts(X)       # 모든 형태소들에게 번호를 부여하고(라벨 붙여줌)
tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1            # 라벨링할때 0을 안쓰기 때문에 1을 더해준다.
print(tokened_x[0:3])
print(wordsize)

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max)
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2 )
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling_data/news_data_max{}_wordsize_{}'.format(max, wordsize), xy)
# 여기까지 데이터 전처리 완료