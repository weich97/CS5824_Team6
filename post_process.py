import numpy as np
DATA_FILE = "Serial-mesh32-convalue0.npz"

data = np.load(DATA_FILE)
train_hist = data['a'][()]
print(train_hist['prediction'])
