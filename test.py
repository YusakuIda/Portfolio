import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD

DATADIR = "/mnt/c/Users/imoga/ac/media2/data"
CATEGORIES = ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2","3_3"]
IMG_SIZE = 224
training_data = []

def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass

def select_label(y_test):
    if y_test == 0:
        return '0'
    elif y_test == 1:
        return '1_2'
    elif y_test == 2:
        return '1_3'
    elif y_test == 3:
        return '2_1'
    elif y_test == 4:
        return '2_2'
    elif y_test == 5:
        return '2_3'
    elif y_test == 6:
        return '3_1'
    elif y_test == 7:
        return '3_2'
    else:
        return '3_3'

create_training_data()

random.shuffle(training_data)  # データをシャッフル

X_train = []  # 画像データ
Y_train = []  # ラベル情報

# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    Y_train.append(label)

# numpy配列に変換
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# trainとtestに分類
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0, stratify=Y_train)

# ===== VGG16で事前に学習されたモデルを読み込む
vgg16 = VGG16(
    include_top = False,
    weights = 'imagenet',
    input_shape = (224, 224, 3)
)

# ===== 新しい出力用分類モデルを作成する
# VGG16の出力を平坦化して全結合層へ接続
x = layers.Flatten()(vgg16.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
# 分類のための10ノードに接続
outputs = layers.Dense(9, activation='softmax')(x)

# モデルの作成
model = keras.Model(inputs=vgg16.input, outputs=outputs)

# 0~18までのlayerがVGG16に関連するので重みを固定する
for layer in model.layers[:19]:
    layer.trainable = False

# モデル構成の表示&画像保存
print(model.summary())
#keras.utils.plot_model(
    #model,
    #'transfer_learning_vgg16_cifar10.png',
    #show_shapes=True
#)

# ===== オプティマイザ、損失関数、指標を設定してコンパイル
# 転移学習の場合、最適化関数はSGDの選択がよいとされている
model.compile(
    optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===== fitを使ったモデルの訓練
# 設定
num_epochs = 100
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'vgg16_cifar10.keras', save_best_only=True
    )
]
# 訓練の実行
history = model.fit(
    x_train,
    y_train,
    epochs=num_epochs,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# ===== history情報の可視化
# 損失関数(loss)の履歴
loss = history.history['loss']
val_loss = history.history['val_loss']
# 正解率(accuracy)の履歴
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# 損失関数の履歴描画
x_epoch = range(1, num_epochs + 1)
plt.plot(x_epoch, loss, 'r', label='training loss')
plt.plot(x_epoch, val_loss, 'b', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# 正解率の履歴描画
plt.figure()
plt.plot(x_epoch, acc, 'r', label='training acc')
plt.plot(x_epoch, val_acc, 'b', label='validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("test1.jpg")

#testを用いた評価
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
print(probability_model(x_test[:4]))
#test画像を表示4個
for i in range(0, 8):
    print("学習データのラベル：", y_test[i])
    plt.subplot(4, 4, i+1)
    plt.axis('off')
    plt.title(label = select_label(y_test[i]))
    plt.imshow(x_test[i])
plt.savefig("test.jpg")

# ===== evaluateを使ったテストデータでの評価
#result = model.evaluate(x_test, y_test)
#print(result)
# ===== predictを使って予測結果を表示
#preds = model.predict(x_test)
#print(f'予測: {np.argmax(preds[0])}, 正解: {y_test[0]} (- 1)')

model.save_weights('./checkpoints/checkpoint')
model.save('saved_model/my_model.h5')