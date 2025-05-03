import pandas as pd
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

df_size = pd.read_csv("./img_folder_size.csv")

# 説明変数（Numpy配列）（一応、2次元配列という体裁をつける必要があるらしい。）
x = df_size[['UNIXタイムスタンプ']].values
# 目的変数（Numpy配列）
y = df_size['imgフォルダのサイズ'].values

model = LinearRegression()
model.fit(x, y)

print('係数:', model.coef_[0])
print('切片:', model.intercept_)

# 1ヶ月後のUNIXタイムスタンプを計算
new_unix = [[int(time.time()) + 2592000]]
# GB単位で未来のサイズを予測
size_pred = model.predict(new_unix)[0] / (10 ** 9)
size_pred = np.round(size_pred, decimals=3)
print("1ヶ月後のimgフォルダのサイズ予測:", size_pred, 'GB')

plt.scatter(x, y, color='blue')
# 回帰直線をプロット
plt.plot(x, model.predict(x), color='red')
plt.xlabel('UNIX timestamp')
plt.ylabel('Size of img folder [B]')
plt.grid()
plt.show()
