import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#initial values and the model
x = 4
a=13
b=12
c=20
y=(x*x*x*x+x*x*x-a*x*x-x+b)/c

#parameters of the gradient descent algorithm
eta = 0.001 # step size multiplier
precision = 0.00001
step_size = abs(x)

#variables and data used in visualization (plots)
n = 200
xn = np.linspace(-4,4,n)
xn = np.array(xn).astype(np.float32)
yn = (xn*xn*xn*xn+xn*xn*xn-a*xn*xn-xn+b)/c

#set font size for the plot and configure the plot
matplotlib.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(10,8), dpi=96)
p1=fig.add_subplot(111)
p1.set_xlabel('x')
p1.set_ylabel('y')
p1.plot(xn,yn,'k')
makersize=28;

#draw the marker for initial x/y values
p1.plot(x, y, marker='o', markersize=makersize, markerfacecolor="none", color="red")

#gradient (derivative) of the y function
def df(x):
    return (4*x*x*x+3*x*x-26*x-1)/20

i=0

#iterative algorithm
while step_size > precision:
    x_p = x
    x =x - eta * df(x_p)
    print("Actual x %f\n" % x)
    step_size = abs(x - x_p)
    y = (x * x * x * x + x * x * x - a * x * x - x + b) / c
    i+=1;
    if makersize>4:
        makersize=makersize-2;
    p1.plot(x, y, marker='o', markersize=makersize, markerfacecolor="none", color="red")

print("The final local minimum of y is %f and occurs at %f" % (y,x))
print("Number of steps %f" % (i))

#plot final result
p1.plot(x, y, marker='x', markersize=20, markerfacecolor="none",color="blue")
plt.show()