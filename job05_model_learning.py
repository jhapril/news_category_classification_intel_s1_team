import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('./crawling_data/news_data_max21_wordsize_12458.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 모델 생성
model = Sequential()
model.add(Embedding(12458, 300, input_length=21))      # 임베딩레이어 - 자연어 학습시 매우 중요. 의미공간상의 벡터 레이어링을 해준다.
# 300 -> 차원 축소. 12000개를 300개의 차원으로 줄인다.
# 공간이 커질수록 데이터갯수는 그대로. 차원이 늘어날수록 차원 안의 데이터가 희소해진다. 데이터가 부족해서 학습이 안됨->차원의 저주
# 차원을 줄여줘야한다.
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))     # Conv1D -> 문장은 한줄이므로.
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))   # return~ 결과값을 하나하나 저장해서 sequential한 데이터로 출력해준다. 이게 없으면 맨 마지막데이터만 출력
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()