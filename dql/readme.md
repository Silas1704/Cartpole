### Cartpole Model USing DQN
## This repo focuses on creating a Deep Q Learing model for the purpose of training a cartpole to stand still without falling down.

### To Run 
``` bash
python train_cartpole.py
```

## In case you get the __version__ error . Go to that path and,
```bash
#change
#from tensorflow.keras import __version__ as KERAS_VERSION
#to
from keras import __version__

```