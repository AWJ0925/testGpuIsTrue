# GPU

## Pytorch

```python
# 测试够是否可用
print("GPU is available ?", torch.cuda.is_available())

# 设备类型的设置
device = torch.device("cuda:0" if torch.cuda.is_available() "cpu:0")

device = torch.device("cuda", 0)
device = torch.device("cpu:0")
```

## TensorFlow

```python 
# 测试GPU是否可用
print('GPU is available ?', tf.test.is_gpu_available())

# TensorFlow中使用GPU
https://www.cnblogs.com/zingp/p/12315366.html

# 禁用GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
   
```
