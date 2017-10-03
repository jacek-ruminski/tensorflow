#A simple polynomial regression example

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Create 100  (x, y) data points in NumPy, y = x1 * 0.1 +x2*0.2+ 0.3
x_training = np.random.rand(100,2).astype(np.float32)
W_training=np.array([0.1,0.2])
B_bias=0.3
y_training = np.dot(x_training,W_training.T)+B_bias
print("X_train=")
print(x_training)
print("\nW_train=")
print(W_training)
print("\nY_train=")
print(y_training)

# Try to find values for W and b for y=W*x+b
# (We know that W should be [0.1, 0.2] and b 0.3, but TensorFlow should estimate the parameters


# 1. First set initial values for W and b; and calculate y
n=2;
W = tf.Variable(tf.random_uniform([n,1], -1.0, 1.0))
print("\nW=")
print(W)
b = tf.Variable(tf.zeros([1]))
print("\nb=")
print(b)


y = tf.add(tf.matmul(x_training,W),b)
print("\ny=")
print(y)

# 2. Define the loss function - e.g. minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_training))

# 3. Define the optimizer - how the training (learning) will be implemented
optimizer = tf.train.GradientDescentOptimizer(0.05)
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
        print("\nLoss: ", sess.run(loss))

print("\nFinal loss: ", sess.run(loss))

fig=plt.figure(1)
ax = fig.add_subplot(1, 1, 1, projection='3d')

#p = ax.plot_surface(x_training[:, 0].T, x_training[:, 1].T, y_training, rstride=1, cstride=1, linewidth=0)
p = ax.plot_wireframe(x_training[:, 0].T, x_training[:, 1].T, y_training, rstride=4, cstride=4)

plt.show()