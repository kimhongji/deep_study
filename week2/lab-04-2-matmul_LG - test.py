# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor that will be always fed.
X = tf(tf.float32, shape=[3, 5])
Y = tf(tf.float32, shape=[1, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer()
train = optimizer.minimize()

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", )

'''
0 Cost:  7105.461990           Cost:  3.18588
Prediction:                    Prediction:
 [[ 80.82241058]                [[ 154.36352539]
 [ 92.26364136]                  [ 182.94833374]
 [ 93.70250702]                  [ 181.85189819]
 [ 98.09217834]                  [ 194.35585022]
 [ 72.51759338]]                 [ 142.03240967]]
10 Cost:  5.89726              2000 Cost:  3.1781
Prediction:                    Prediction:
 [[ 155.35159302]              [[ 154.35881042]
 [ 181.85691833]                [ 182.95147705]
 [ 181.97254944]                [ 181.85035706]
 [ 194.21760559]                [ 194.35533142]
 [ 140.85707092]]               [ 142.036026  ]]

...
