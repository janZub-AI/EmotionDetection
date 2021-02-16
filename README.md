# EmotionDetection

A little bit of experimenting and exploring networks. Achived a little under 65% on test set for FER-2013.

After 65th (last recorded) epoch:
|Hyperparameter | Value|
---|---
init              |96
cnn_layers        |2
cnn_1             |224
cnn_2             |224
pooling           |
cnn2_layers       |2
cnn2_1            |160
cnn2_2            |160
pooling           |
dense             |1000
dropout           |0.5
dense2            |300
dropout           |0.5
learning_rate     |0.001

Evaluate on test data

>760/760 - [##################] - 15s 19ms/step - loss: 0.0039 - accuracy: 0.9983  
139/139  - [##################] - 3s 19ms/step - loss: 2.4796 - accuracy: 0.6298
225/225  - [##################] - 4s 19ms/step - loss: 2.4694 - accuracy: 0.6464  

dev set was created from 0.1 of training set, should have use part of test

>train loss, train acc: [0.00391763262450695, 0.9983121752738953]
dev loss, dev acc: [2.4796061515808105, 0.6298392415046692]
test loss, test acc: [2.4694037437438965, 0.6464196443557739]
