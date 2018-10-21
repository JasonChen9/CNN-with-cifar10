# 使用cifar10数据集训练进阶神经网络

进阶的主要区别在于：

- 对weights进行了L2的正则化

- 对原始数据进行处理，制造了更多样本

- 在每个卷积-最大池化层后面使用了LRN层，增强了模型的泛化能力


下面来具体看代码：

首先导入一些必要的库和数据

```python
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
```

然后设置一些基本的参数，包括最大迭代次数，一个batch的大小和数据的存储路径

```python
max_steps= 3000
batch_size =128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
```

然后是weight函数，这里面设置了一个w1参数，如果w1不为0就对初始化的权重进行L2的正则化并用w1控制L2loss的大小

```python
def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))#使用截断的正态分布来初始化权重
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')#做l2的正则化，用w1来控制l2 loss的大小
        tf.add_to_collection('losses',weight_loss)#把结果保存在一个collection
    return var
```

之后我们来加载数据：

```python
cifar10.maybe_download_and_extract()#使用cifar10类下载数据集，并解压，展开到默认位置

images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)#生成测试数据
```

设置占位符：

```python
image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])
```



由于我们设置好了batch_size的大小所以在设置占位符的时候不能再把shape设置为none

下面来建立神经网络：

```python
#第一层
weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)#第一层卷积核使用5乘5的卷积核输入通道为3输出通道为64 不使用l2正则化因此w1设为0
kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')#卷积
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))#设置偏置
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))#激活
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#最大池化
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)#局部响应归一化
    #第二层
weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)#输入通道64输出通道64
kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')#卷积
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))#设置偏置
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))#激活
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)#局部响应归一化
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#池化
    #调换了lrn与池化的顺序
    #设置两层全连接层
reshape = tf.reshape(pool2,[batch_size,-1])#将函数每个样本都变为一维向量
dim = reshape.get_shape()[1].value#获得数据扁平化后的值
weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)#设置权重
bias3 =tf.Variable(tf.constant(0.1,shape=[384])) #设置偏置
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)
    #最后一层
weight5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5),bias5)
```

定义损失函数：

```python
def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)#转换数据类型
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,labels=labels,name='cross_entropy_per_example'
    )#交叉熵加入了softmax计算
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')#求平均
    tf.add_to_collection('losses',cross_entropy_mean)#把损失加到collection里
    return tf.add_n(tf.get_collection('losses'),name='total_loss')
```



训练模型：

```python
loss=loss(logits,label_holder)#得到损失
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)#定义训练，学习速率为1e-3最小化loss
top_k_op = tf.nn.in_top_k(logits,label_holder,1)#得到概率最高的预测值与标签相同的正确率
sess = tf.InteractiveSession()#创建默认的session
sess.run(tf.initialize_all_variables())#初始化变量
tf.train.start_queue_runners()#启动线程队列
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch=sess.run([images_train,labels_train])
    _,loss_value = sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration = time.time()-start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str=('step %d,loss=%.2f (%.1f example/sec;%.3f sec/batch)')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
```



测试模型得到准确率：

```python
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))#返回大于等于参数的最小整数
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count += np.sum(predictions)#正确的个数
    step += 1
precision = true_count / total_sample_count#正确率等于正确的个数除总个数
print('precision @ 1=%.3f'% precision)
```

由于手上gpu计算能力有限并没有跑完整个迭代过程，不过测试了30次迭代后认为模型应该没有bug



## 总结

本次进阶模型比起简单神经网络模型加入了正则化，噪音和局部响应归一化。

正则化避免过拟合，由于神经网络的参数非常多，不加限制的拟合参数可能会使模型很好的拟合训练集但没有很好的泛化能力，而加入了正则化则是限制了权重值，使那些不重要的权重尽可能小并且重要的权重也保留了下来，这样模型会得到比较好的泛化能力。

对原始数据的预处理增加了样本数，并且在原始数据中加入了噪音，也有利于提高模型的泛化能力。

局部响应归一化（LRN)：局部归一的动机：在神经生物学有一个概念叫做侧抑制，指的是被激活的神经元抑制相邻神经元。归一化的的目的就是“抑制”，局部响应归一化就是借鉴侧抑制的思想来实现局部抑制，尤其当我们使用ReLU的这种“侧抑制”很管用。好处：有利于增加泛化能力，做了平滑处理，识别率提高1-2%。LRN层模仿生物神经系统的侧抑制机制，对局部神经元的活动创建竞争机制，使得响应比较大的值相对更大，提高模型泛化能力。一一般参数设置为 k=2,n=5,aloha=1e-4 beta=0.75

