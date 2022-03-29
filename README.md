## Homework__Git和Github使用体会
<br/>
这里是git的下载链接
<br/>
[Git的下载链接](https://git-scm.com/)


#### 大创的部分代码
```javascript
from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
df = pd.read_excel('C:/Users/hp/Desktop/Dataset/sujing1.xlsx')
data_all=df.values
#数据集划分
data = data_all[:1800]
test = data_all[1800:]
#数据预处理(r==1)标签one-hot编码(r==0)
def vectorize_sequences(sequences, dimension=360,r=0):
    if(r==1):
        results = np.zeros((len(sequences), dimension))
        for i in range(len(sequences)):
            for j in range(4):
                results[i, sequences[i][j]] = 1.
    else:
        results = np.zeros((1, dimension))
        for j in range(4):
            results[0, sequences[j]] = 1.
    return results
#对训练数据和测试数据进行预处理
x_train = vectorize_sequences(data,r=1)
x_test = vectorize_sequences(test,r=1)
def to_one_hot(labels, dimension=4):
    results = np.zeros((len(labels),dimension))
    for i in range(len(labels)):
        results[i, labels[i][4]] = 1.
    return results
#对训练数据标签和测试数据标签进行one-hot编码
one_hot_train_labels = to_one_hot(data)
one_hot_test_labels = to_one_hot(test)
#网络层构建
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(360,)))
model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(48,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
              metrics=['accuracy'])
x_val = x_train[:200]
partial_x_train = x_train[200:]
y_val = one_hot_train_labels[:200]
partial_y_train = one_hot_train_labels[200:]
history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=100,validation_data=(x_val,y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
#绘制损失曲线图
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
#绘制精度曲线图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#模型训练
results = model.evaluate(x_test,one_hot_test_labels)
print(results)
#模型存储
model.save('C:/Users/hp/Desktop/Dataset/my_model.h5')
input = np.array([72,14,200,250])
f = model.predict(vectorize_sequences(input))
out = np.argmax(f, axis=1)
print(out)
#os.system('C:/Users/hp/Desktop/Dataset/0.wav')

```
#### 如果你希望嵌入一张图片，可以这么做：
![2021年杏花节图片](https://github.com/zhj0415/Homework/blob/main/Xinghua.JPG)
![网络图片](https://octodex.github.com/images/dinotocat.png)


- First item
- Second item
- Third item
 - Indented item
 - Indented item
- Fourth item

1. First item
2. Second item
3. Third item
4. Fourth item

姓名 | 学号
------------ | -------------
张恒嘉 | 21190514
xxx | 2119xxxx
<br/>
使用 Markdown 可以非常容易地为一些单词设置 **加粗** 而为另外一些单词设置 *倾斜*。
<br/>
~~ 删除 ~~
[跳转到other.md](./other.md)
<br/>



