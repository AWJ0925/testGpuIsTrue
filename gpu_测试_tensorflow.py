# 0.GPU是否可用
import tensorflow as tf
print("GPU是否可用：" + str(tf.test.is_gpu_available()))

# 1.查看本机GPU/CPU信息：
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 2.指定运行设备名称：
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

# 3.查看当前加载的设备：
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
