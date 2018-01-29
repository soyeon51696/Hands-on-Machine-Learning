
# CHAPTER.10 Introduction to Artificial Neural Networks




```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
```

# 1.	From Biological to Artificial Neurons


## 1.1	Biological neurons

## 1.2	Logical Computations with Neurons

## 1.3	The Perceptron


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
```

    /home/soyeon/sy/lib/python3.5/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.perceptron.Perceptron'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)



```python
y_pred
```




    array([1])



## 1.4	Multi-Layer Perceptron and Backpropagation


```python
def logit(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)
```


```python
z = np.linspace(-5, 5, 200)

plt.figure(figsize=(11,4))

plt.subplot(121)
plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step")
plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
plt.legend(loc="center right", fontsize=14)
plt.title("Activation functions", fontsize=14)
plt.axis([-5, 5, -1.2, 1.2])

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
plt.grid(True)
#plt.legend(loc="center right", fontsize=14)
plt.title("Derivatives", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])

save_fig("activation_functions_plot")
plt.show()
```

    Saving figure activation_functions_plot



![png](output_10_1.png)


# 2.	Training an MLP with Tensorflow¡¯s High-Level API


```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
```

    /home/soyeon/sy/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-4141630e56b4> in <module>()
          1 from tensorflow.examples.tutorials.mnist import input_data
          2 
    ----> 3 mnist = input_data.read_data_sets("/tmp/data/")
    

    ~/sy/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py in read_data_sets(train_dir, fake_data, one_hot, dtype, reshape, validation_size, seed, source_url)
        238 
        239   local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
    --> 240                                    source_url + TRAIN_IMAGES)
        241   with gfile.Open(local_file, 'rb') as f:
        242     train_images = extract_images(f)


    ~/sy/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py in maybe_download(filename, work_directory, source_url)
        206   filepath = os.path.join(work_directory, filename)
        207   if not gfile.Exists(filepath):
    --> 208     temp_file_name, _ = urlretrieve_with_retry(source_url)
        209     gfile.Copy(temp_file_name, filepath)
        210     with gfile.GFile(filepath) as f:


    ~/sy/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py in wrapped_fn(*args, **kwargs)
        163       for delay in delays():
        164         try:
    --> 165           return fn(*args, **kwargs)
        166         except Exception as e:  # pylint: disable=broad-except)
        167           if is_retriable is None:


    ~/sy/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py in urlretrieve_with_retry(url, filename)
        188 @retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
        189 def urlretrieve_with_retry(url, filename=None):
    --> 190   return urllib.request.urlretrieve(url, filename)
        191 
        192 


    /usr/lib/python3.5/urllib/request.py in urlretrieve(url, filename, reporthook, data)
        215 
        216             while True:
    --> 217                 block = fp.read(bs)
        218                 if not block:
        219                     break


    /usr/lib/python3.5/http/client.py in read(self, amt)
        446             # Amount is given, implement using readinto
        447             b = bytearray(amt)
    --> 448             n = self.readinto(b)
        449             return memoryview(b)[:n].tobytes()
        450         else:


    /usr/lib/python3.5/http/client.py in readinto(self, b)
        486         # connection, and the user is reading more bytes than will be provided
        487         # (for example, reading in 1k chunks)
    --> 488         n = self.fp.readinto(b)
        489         if not n and b:
        490             # Ideally, we would raise IncompleteRead if the content-length


    /usr/lib/python3.5/socket.py in readinto(self, b)
        573         while True:
        574             try:
    --> 575                 return self._sock.recv_into(b)
        576             except timeout:
        577                 self._timeout_occurred = True


    /usr/lib/python3.5/ssl.py in recv_into(self, buffer, nbytes, flags)
        927                   "non-zero flags not allowed in calls to recv_into() on %s" %
        928                   self.__class__)
    --> 929             return self.read(nbytes, buffer)
        930         else:
        931             return socket.recv_into(self, buffer, nbytes, flags)


    /usr/lib/python3.5/ssl.py in read(self, len, buffer)
        789             raise ValueError("Read on closed or unwrapped SSL socket.")
        790         try:
    --> 791             return self._sslobj.read(len, buffer)
        792         except SSLError as x:
        793             if x.args[0] == SSL_ERROR_EOF and self.suppress_ragged_eofs:


    /usr/lib/python3.5/ssl.py in read(self, len, buffer)
        573         """
        574         if buffer is not None:
    --> 575             v = self._sslobj.read(len, buffer)
        576         else:
        577             v = self._sslobj.read(len)


    KeyboardInterrupt: 



```python
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")
```


```python
import tensorflow as tf

config = tf.contrib.learn.RunConfig(tf_random_seed=42) # not shown in the config

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                         feature_columns=feature_cols, config=config)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf) # if TensorFlow >= 1.1
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
```


```python
from sklearn.metrics import accuracy_score

y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test, y_pred['classes'])
```

# 3.	Training a DNN Using Plain Tensorflow


```python
import tensorflow as tf

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
```


```python
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
```


```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
```


```python
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
```


```python
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
```


```python

# with tf.name_scope("dnn"):
  #  hidden1=tf.layers.dense(X,n_hidden1, name="hidden1", activation=tf.nn.relu)
  #  hidden2=tf.layers.dense(hidden1, n_hidden2, name="hidden2",activation=tf.nn.relu)
  #  logits=tf.layers.dense(hidden2, n_outputs, name="outputs")
    

```


```python
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
```


```python
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
```


```python
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```


```python
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```


```python
n_epochs = 40
batch_size = 50
```


```python
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                            y: mnist.validation.labels})
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```


```python
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
```


```python
print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])
```
