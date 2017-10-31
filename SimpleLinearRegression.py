#A simple linear regression example

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate 100 random data points for the x variable
n_samples=100
x_t = np.random.rand(n_samples).astype(np.float32)
# sort data - easier visualization only
x_training = np.sort(x_t)


# Assume that our true description of the phenomena is
# y = 0.2* x  + 0.4 + noise

# First assume that there is no noise but in next experiment we can include it
#noise=0
noise=np.random.rand(n_samples).astype(np.float32)/20
Wt=0.2
bt=0.4
y_training = Wt * x_training  + bt +noise

print("\n Noise=", np.std(noise))

# We can print our data
print("\n x_training=", x_training)
print("\n y_training=", y_training)

#f=plt.figure(figsize=(8,4), dpi=96)
#f1=f.add_subplot(111)
#f1.set_xlabel('x')
#f1.set_ylabel('y (true)')
#f1.plot(x_training,y_training,'k')

# Now we would like to estimate the "unknown" parameters of the assumed model
# Find W and b for y=W*x+b

# 1. First we need to define "a place" where the estimated values of the parameters
# will be stored (the values are changing during learning). Therefore, we will use VARIABLES
# Additionally, we will set initial values (starting values, a guess) for the parameters
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
# The assumed model
y = W * x_training + b


# 2. We need to define the loss function - e.g. minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_training))
loss_values=[]

# 3. We need to define the optimizer - how the training (learning) will be implemented
learnig_rate=0.01
optimizer = tf.train.GradientDescentOptimizer(learnig_rate)
train = optimizer.minimize(loss)

# 4. Initialize all variables in TensorFlow
init = tf.global_variables_initializer()

# 5. All is ready - define the session and run the initialization process
sess = tf.Session()
sess.run(init)

# 6. Now do the training, but do it in many trails (epochs)
epochs=4000
for step in range(epochs):
    sess.run(train)
    loss_values.append(sess.run(loss))
    #print results for some iterations
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

print("\nFinal loss: ", sess.run(loss))
print("\nParameters:", sess.run(W), sess.run(b))

# Calculate a simple difference between real data and the model
diff=y_training-sess.run(y)

# Show results using plots
fig=plt.figure(figsize=(10,8), dpi=96)
fig.suptitle('True: y=%sx+%s(+noise). Learned: y=%sx+%s.'%(Wt,bt,sess.run(W), sess.run(b)),
             fontsize=14, fontweight='bold')

p1=fig.add_subplot(411)
p1.set_xlabel('x')
p1.text(0.95, 0.01, 'x',verticalalignment='bottom', horizontalalignment='right', transform=p1.transAxes)
p1.set_ylabel('y (true)')
p1.plot(x_training,y_training,'k')

p2=fig.add_subplot(412)
p2.text(0.95, 0.01, 'x',verticalalignment='bottom', horizontalalignment='right', transform=p2.transAxes)
p2.set_ylabel('y (model)')
p2.plot(x_training,sess.run(y),'b')

p3=fig.add_subplot(413)
p3.text(0.95, 0.01, 'x',verticalalignment='bottom', horizontalalignment='right', transform=p3.transAxes)
p3.set_ylabel('difference')
p3.plot(diff,'r')

p4=fig.add_subplot(414)
p4.set_xlabel('epoch')
p4.set_ylabel('loss')
p4.plot(loss_values,'m')

plt.show()