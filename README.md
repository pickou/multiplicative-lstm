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

train_gentext.py result, the text make sense.
```
('Training epoch: 2/2***', 'Training steps: 500***', 'Training loss: 1.6599***', '0.8138 sec/per batch')
This morning how
so menty there a dingry was not be their conversnon had not been to to hor that they was not had been so would bating the sout. All at her to her some house, bown sacking at the saces and said of
 his
witches of the propission his hand of his husband.

"Well, all which his was as it as answained to and he saw. A drovin and with the money
have to delight.

"Yes, this the simply. That he w
```
load_gentext.py result, make sense.

```
This morning what who saw they wife to this. There and allet her tomer as his weatther to stall that the consed and sack that the bagh falt of
this who stept,
were to her ever were a troube, and sack that it was the sawe his heart, and that should
not that I would not see that it was it a to be
done when the counse, and a carting thought he could not
could not her his hadis with the mander of at a still
```

+ using python3 to run train_gentext.py and load_gentext.py

load_gentext.py result, make no sense.
```
This morningo(@4g3qg@T:S(@4RsSn@?:@(:;:T@(g:Wb@g:@To3n,%I%IvO@4o*(@o@SR*Z@o(@(g:@mR;(@o*n@4:*(@R'@(g:@W3T(:;@oT@g:%Igon@*R(%Io*n@4RsSn@*R(%I(;o(gTg@o@W3Zg(@o@q:*(;o(@gon%I?oq9@o(@o@T:*(@o*n@(g:@q
RSK:TT@(go(@gR4@n:T:n@(R@(o9:@g3T@4RsSn@goK:@o@To4@(g:W@o*n@4:*(@R*:@gon@?::*@(R@n;:K:*@(R@(g:@T(3SS@?s(@g3T@4o;n:n%I(g:@moTTb@(go(@(g:;3q:@R'@3*@43(gRs(@o@TRs*n:n@o*n@4gR@T::W@g3T@'3*Z:;b@g:%In3
n@*R(@qo;(@3*@(g:@T:*(@
```

+ using python2 to run train_gentext.py  and python3 to run load_gentext.py

The result is the same as the above method using python3 for two .py.

I've checked the checkpoint file using checkpoint reader, the variable and value is right.
So I think the bug lies in restore in python3, python2 is fine.
