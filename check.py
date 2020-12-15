import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")

print(f"GPU available: {tf.test.is_gpu_available()}")

print(f"GPU device name: {tf.test.gpu_device_name()}")

print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
