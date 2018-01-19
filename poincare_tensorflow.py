import tensorflow as tf
import numpy as np 
from math import sqrt
STABILITY = 0.00001

def my_partial_der(theta, x):
	alpha = 1 - np.dot(theta, theta)
	beta = 1 - np.dot(x, x)
	gamma = 1 + 2.0 / (alpha * beta) * np.dot(theta - x, theta - x)

	return 4.0/(beta * sqrt(gamma*gamma - 1.0)) * ((np.dot(x,x) - 2 * np.dot(theta, x) + 1) / (alpha*alpha) * theta - x/alpha)

def partial_der(theta, x): #eqn4
    alpha = 1.0-np.dot(theta, theta)
    norm_x = np.dot(x, x)
    beta = 1-norm_x
    gamma = 1.0 + 2.0 / (alpha * beta) * np.dot(theta - x, theta - x)



    print 'alpha: ' ,alpha
    print 'beta: ', beta
    print 'gamma: ', gamma

    return 4.0/(beta * sqrt(gamma*gamma - 1) + STABILITY)*((norm_x- 2*np.dot(theta, x)+1)/(pow(alpha,2)+STABILITY)*theta - x/(alpha + STABILITY))

def mobius_addition_numpy(u, v):

	numerator = np.add(np.multiply(1 + 2 * np.dot(u,v) + np.dot(v,v), u),np.multiply(1 - np.dot(u,u), v))
	denominator = 1 + 2 * np.dot(u,v) + np.dot(u,u) * np.dot(v,v)
	return np.divide(numerator, 1.0 * denominator)
	"""
	numerator = (1 + 2 * np.dot(u,v) + np.dot(v,v)) * u + (1 - np.dot(u,u)) * v
	denominator = 1 + 2 * np.dot(u,v) + np.dot(u,u) * np.dot(v,v)

	print 'Numerator: ', numerator
	print 'Denominator: ', denominator
	return numerator / denominator
	"""

def distance_poincare(u,v):
	u = np.array(u)
	v = np.array(v)
	return 1 + 2 * np.dot(u - v, u - v) / ((1 - np.dot(u,u)) * (1 - np.dot(v,v)))

def dot_product(x, y):
	return tf.reduce_sum(tf.multiply(x,y))


u = tf.placeholder(tf.float32)
v = tf.placeholder(tf.float32)

u_instance = np.random.rand(10)#np.array([1.0,2.0, 3643])
v_instance = np.random.rand(10)#np.array([3.0,4.0, 5.15])

print 'u:', u_instance
print 'v:', v_instance

dist_poincare = tf.acosh(1 + 2 * dot_product(u-v, u - v) / ((1 - dot_product(u,u))*(1 - dot_product(v,v))))

numerator = (1 + 2 * dot_product(u,v) + dot_product(v,v)) * u + (1 - dot_product(u,u)) * v
denominator = 1 + 2 * dot_product(u,v) + dot_product(u,u) * dot_product(v,v)
mobius_addition = tf.divide(numerator, denominator)

gradient_poincare_distance_u, gradient_poincare_distance_v = tf.gradients(ys = dist_poincare, xs = [u,v])


with tf.Session() as sess:
	print 'Gradients:'
	print sess.run(gradient_poincare_distance_u, feed_dict = {
		u: u_instance, 
		v: v_instance
	})
	print sess.run(gradient_poincare_distance_v, feed_dict = {
		u: u_instance, 
		v: v_instance
	})

	print partial_der(theta = u_instance, x = v_instance)
	print my_partial_der(theta = u_instance, x = v_instance)
"""





with tf.Session() as sess:

	print sess.run(mobius_addition, feed_dict = {
		u: u_instance,
		v: v_instance
	})

	print mobius_addition_numpy(u_instance, v_instance)
"""


