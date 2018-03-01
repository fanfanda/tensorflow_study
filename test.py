import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))
x = tf.placeholder(tf.float32,name="x-input")
y_ = tf.placeholder(tf.float32,name="y-input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
#define loss-function
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size,2)
Y = [int(x1+x2 < 1) for (x1,x2) in X]

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(w1))
  print(sess.run(w2))
  for i in range(5000):
    start = (i*batch_size) % data_size
    end = min(start+batch_size,data_size)
    sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
    
    if not i % 1000:
      total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
      print("After %d training steps,cross entropy on all data is %g" % (i,total_cross_entropy))

  print(sess.run(w1))
  print(sess.run(w2))