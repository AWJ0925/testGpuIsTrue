import tensorflow as tf
import os
import time


# 禁用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 使用GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print('GPU is available ? ', tf.test.is_gpu_available())


# 获取数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model: fully-connected net
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation='softmax')])
# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print("-----------------CPU----------------")
# t0 = time.time()
# model.fit(x_train, y_train, epochs=50)
# model.evaluate(x_test, y_test, verbose=2)
# t1 = time.time()
# print("cpu：" + str(t1 - t0))

print("-----------------GPU1----------------")
t1 = time.time()
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test, verbose=2)
t2 = time.time()
print("gpu1：" + str(t2 - t1))

#
print("-----------------GPU2----------------")
model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test, verbose=2)
t3 = time.time()
print("gpu2：" + str(t3 - t2))
# cpu：79.47657585144043
# gpu2：312.5358350276947
# gpu2：308.8298239707947
