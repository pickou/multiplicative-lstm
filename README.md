#  An implementation of mLSTM
This code exposes the bug in tensorflow while restore graph variables in python3. Tensorflow version 1.2.1
### 1. file desicription
+ MultiplicativeLSTMCell.py

The core is written by tam17aki, 
https://github.com/tam17aki/multiplicative-lstm-tensorflow

His implementation is based on the following paper:

Ben Krause, Liang Lu, Iain Murray, and Steve Renals, "Multiplicative LSTM for sequence modelling, " in Workshop Track of ICLA 2017, https://openreview.net/forum?id=SJCS5rXFl&noteId=SJCS5rXFl
+ mLSTM.py

This is an implementation which uses mLSTM to predict the next character of a sentence.

+ train_gentext.py

Train and generate text with the starting sentence "This morning", the restore process is right after training.

+ load_gentext.py

Load checkpoint and restore parameters, then generate text.

### 2. result

+ using python2 to run train_gentext.py and load_gentext.py

```

```

+ using python3 to run train_gentext.py and load_gentext.py
```
```

+ using python2 to run train_gentext.py  and python3 to run load_gentext.py

The result is the same as the above method using python3 for two .py.

I've checked the checkpoint file using checkpoint reader, the variable and value is right.
So I think the bug lies in restore in python3, python2 is fine.
