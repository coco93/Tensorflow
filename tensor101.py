import tensorflow as tf


node1 = tf.constant(3.0, tf.float32)  #can provide value and type
node2 = tf.constant(4.0)   #can provide only value
print(node1, node2)
sess = tf.Session()
print(sess.run([node1,node2]))


#TensorBoard
import tensorflow as tf
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a,b, name="multiply_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")
sess = tf.Session()
output = sess.run(e)
#writer = tf.train.SummaryWriter('./my_graph', sess.graph)  #this has been depracated
writer = tf.summary.FileWriter('./my_graph', sess.graph)  #replaced with this instead
writer.close()
sess.close()

#train linear model
W = tf.Variable([.3], tf.float32)  #weight value
b = tf.Variable([-.3], tf.float32)  #bias value
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# variables need to be initialized before running through a session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
# to have a learning phase, we usually need to add a loss function or an error function

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) #measures how far off every entry in the linear model
loss = tf.reduce_sum(squared_deltas) #sum how far off all entries to have a flat value
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) #result would say how far are we to the correct answer
#answer: 23.66


##if we don't know the correct weight and bias? how do we use machine learning to find
## out the correct weight and bias?

##we're going to use an optimizer. a common optimizer is Gradient Descent
##it basically tweaks the result and see if the loss function is less over and over

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)  #initialize variables
# reset values to incorrect defaults
for i in range(1000):  #iterate 1000 times
    sess.run(train, {x:[1,2,3,4],
                     y:[0,-1,-2,-3]})
print(sess.run([W,b]))
