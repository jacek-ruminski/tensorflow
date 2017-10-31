#A simple linear regression example

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Generate 100 random data points for the x variable
dataset_training_size = 1000
W1=0.2
W2=0.8
bt=0.1
n=2
learnig_rate=0.01
epochs=10001

# Noise
noise=0
#noise=np.random.rand(dataset_training_size).astype(np.float32)/20

# Assume that our true description of the phenomena is
# y = Wt* x  + bt + noise
x_t = np.random.rand(2*dataset_training_size).astype(np.float32)
#x_training = (np.sort(x_t)).reshape(dataset_training_size,2)
x_training = x_t.reshape(dataset_training_size,2)



W_training=np.array([W1,W2])
y_t=(np.dot(x_training,W_training.T)+bt+noise).astype(np.float32)
y_training = np.transpose(np.array([y_t]));

# We can print our data
print("\n x_training=", x_training)
print("\n y_training=", y_training)

matplotlib.rcParams.update({'font.size': 16})
f=plt.figure(figsize=(12,12), dpi=96)
f1 = f.add_subplot(1, 1, 1, projection='3d')
f1.scatter(x_training[:, 0].T, x_training[:, 1].T, y_training)
f1.set_xlabel('x1')
f1.set_ylabel('x2')
f1.set_zlabel('y')

# Now we would like to estimate the "unknown" parameters of the assumed model
# Find W and b for y=W*x+b

# 1. First we need to define "a place" where the estimated values of the parameters
# will be stored (the values are changing during learning). Therefore, we will use VARIABLES
# Additionally, we will set initial values (starting values, a guess) for the parameters
W = tf.Variable(tf.zeros([n,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

# Define placeholders to pass data from a source (training) data
x = tf.placeholder(tf.float32, [None,n], name="x")
y = tf.placeholder(tf.float32, [None,1], name="y")

# The assumed model
y_ = tf.add(tf.matmul(x,W),b)

print(y_.dtype, y_.shape)
print(y_training.dtype, y_training.shape)

# 2. We need to define the loss function - e.g. minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_))
loss_values=[]

# 3. We need to define the optimizer - how the training (learning) will be implemented

optimizer = tf.train.GradientDescentOptimizer(learnig_rate)
train = optimizer.minimize(loss)

# 4. Initialize all variables in TensorFlow
init = tf.global_variables_initializer()

# 5. All is ready - define the session and run the initialization process
sess = tf.Session()
sess.run(init)
all_feed = { x: x_training, y: y_training }

# 6. Now do the training, but do it in many trails (epochs)
for step in range(epochs):
    sess.run(train, feed_dict=all_feed)
    loss_values.append(sess.run(loss, feed_dict=all_feed))
    #print results for some iterations
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        print("\nLoss: ", sess.run(loss, feed_dict=all_feed))

print("\nFinal loss: ", sess.run(loss, feed_dict=all_feed))


# Calculate simple difference between real data and the model
diff=100*(y_training-sess.run(y_,feed_dict=all_feed))/y_training

# Show results using plots
fig=plt.figure(figsize=(10,8), dpi=96)
fig.suptitle('True: y=%sx1+%sx2+%s(+noise). Learned: y=%sx1+%sx2+%s.'
             %(W1,W2,bt,sess.run(W,feed_dict=all_feed)[0],sess.run(W,feed_dict=all_feed)[1], sess.run(b,feed_dict=all_feed)), fontsize=16, fontweight='bold')

p1=fig.add_subplot(411)
p1.text(0.95, 0.01, 'i',verticalalignment='bottom', horizontalalignment='right', transform=p1.transAxes)
p1.set_ylabel('y (true)')
p1.plot(y_training,'k')

p2=fig.add_subplot(412)
p2.text(0.95, 0.01, 'i',verticalalignment='bottom', horizontalalignment='right', transform=p2.transAxes)
p2.set_ylabel('y (model)')
p2.plot(sess.run(y_,feed_dict=all_feed),'b')

p3=fig.add_subplot(413)
p3.text(0.95, 0.01, 'i',verticalalignment='bottom', horizontalalignment='right', transform=p3.transAxes)
p3.set_ylabel('difference')
p3.plot(diff,'r')

p4=fig.add_subplot(414)
p4.set_xlabel('epoch')
p4.set_ylabel('loss')
p4.plot(loss_values,'m')

plt.show()