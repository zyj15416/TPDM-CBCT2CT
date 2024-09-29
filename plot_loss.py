import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    with open('./train_loss.txt', 'r') as f:
        data_train = f.read().splitlines()
    data_train = np.array(data_train, dtype=np.float32)
    data_train = np.array_split(data_train, 100)
    data_train = np.array(data_train)
    data_train = np.mean(data_train, axis=-1)

    with open('./val_loss.txt', 'r') as f:
        data_val = f.read().splitlines()
    data_val = np.array(data_val, dtype=np.float32)
    data_val = np.array_split(data_val, 100)
    data_val = np.array(data_val)
    data_val = np.mean(data_val, axis=-1)

    plt.figure(figsize=(12, 12))
    plt.plot(np.arange(0, len(data_train)), data_train, label='train_loss')
    plt.plot(np.arange(0, len(data_val)), data_val, label='val_loss')
    plt.legend()
    plt.show()
