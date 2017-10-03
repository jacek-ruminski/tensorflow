#A simple linear regression example

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100  (x, y) data points in NumPy, y = x * 0.1 + 0.3
x_training = np.random.rand(100).astype(np.float32)
y_training = x_training * 0.1 + 0.3
print("\nX_train=")
print(x_training)
print("\nY_train=")
print(y_training)

# Try to find values for W and b for y=W*x+b
# (We know that W should be 0.1 and b 0.3, but TensorFlow should estimate the parameters

# 1. First set initial values for W and b; and calculate y
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print("\nW=")
print(W)
b = tf.Variable(tf.zeros([1]))
print("\nb=")
print(b)
y = W * x_training + b
print("\ny=")
print(y)

# 2. Define the loss function - e.g. minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_training))

# 3. Define the optimizer - how the training (learning) will be implemented
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 4. Initialize all variables in TensorFlow
init = tf.global_variables_initializer()

# 5. All is ready - define the session and run the initialization process
sess = tf.Session()
sess.run(init)

# 6. Now do the training
epochs=301
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

print("\nFinal loss: ", sess.run(loss))

diff=y_training-sess.run(y)

plt.figure(1)
plt.subplot(311)
plt.plot(x_training,y_training,'k')
plt.subplot(312)
plt.plot(x_training,sess.run(y),'b')
plt.subplot(313)
plt.plot(diff,'r--')
plt.show()