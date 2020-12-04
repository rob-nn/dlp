import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from time import time


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension), dtype='float32')
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

print (f"dlp library loaded with tensorflow {tf.__version__}")

def run_model(model, x_val, partial_x_train, y_val, partial_y_train, x_test, y_test):
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

    history_dict = history.history
    
    print ('History keys: ', history.history.keys())
    
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) +1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict['accuracy']
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    val_acc = history_dict['val_accuracy']
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results, t= timeit(lambda : model.evaluate(x_test, y_test))
    print('Evaluation: ', results, t)

    r, t = timeit(lambda: model.predict(x_test))
    print ('Tests predict: ', r, t)
    
def timeit(some_func):
    
    i = time()
    return some_func(), time() -i

