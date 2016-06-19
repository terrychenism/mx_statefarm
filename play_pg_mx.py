""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
from matplotlib import pyplot as plt
import cv2
import mxnet as mx
from sklearn.datasets import fetch_mldata
import logging
from collections import namedtuple

def get_symbol_conv():
    data = mx.sym.Variable(name='x')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, kernel=(5,5), num_filter=5)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                                  kernel=(2,2), stride=(2,2))


    conv2 = mx.symbol.Convolution(name='conv2', data=pool1, kernel=(5,5), num_filter=10)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                                  kernel=(2,2), stride=(2,2))

    flatten = mx.symbol.Flatten(data=pool2)
    fc = mx.sym.FullyConnected(name='fc', data=flatten, num_hidden=1)
    act = mx.sym.Activation(data=fc, act_type='sigmoid')
    return act

FcExecutor = namedtuple('FcExecutor', ['executor', 'data', 'data_grad', 'act'])
def get_symbol_fc(input_size, ctx):
    data = mx.symbol.Variable(name='data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=200)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 200)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=1)
    out = mx.sym.Activation(data=fc3, act_type='sigmoid')
    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(1, input_size))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    print arg_dict
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return FcExecutor(executor=executor,
                        data=arg_dict["data"],
                        data_grad=grad_dict["data"],
                        act=executor.outputs[0])


class RandIter(mx.io.DataIter):
    def __init__(self, cur_x, prev_x, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('x', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return np.array(cur_x-prev_x)

def Softmax(theta):
    max_val = np.max(theta, axis=1, keepdims=True)
    tmp = theta - max_val
    exp = np.exp(tmp)
    norm = np.sum(exp, axis=1, keepdims=True)
    return exp / norm

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) 
  
def LogLossGrad(alpha, label):
    grad = np.copy(alpha)
    print 'alpha.shape: ', alpha.shape[0]
    for i in range(alpha.shape[0]):
        grad[i, label[i]] -= 1.
    return grad


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  gamma = 0.99 # discount factor for reward
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

GAME_TITLE = "Pong-v0"
env = gym.make(GAME_TITLE)
observation = env.reset()
render = True
lr = 1e-4
prev_x = None
xs,hs1,hs2, dlogps,drs = [],[],[],[],[]
running_reward = None
reward_sum = 0
loss = 0
episode_number = 0
ctx = mx.gpu(0)
Z = 6400
beta1 = 0.5
D = 80 * 80
dev = mx.gpu(0)
size = Z
batchsize = 1

# model_executor = get_symbol_fc(size, dev)

data = mx.symbol.Variable(name='data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=300)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 300)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
out  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=1)
# out = mx.sym.Activation(data=fc3,act_type='sigmoid')
# make executor
arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(batchsize, size))
arg_names = out.list_arguments()
arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
print arg_names

arg_dict['fc1_weight'][:] = np.random.normal(0, 0.125, arg_dict['fc1_weight'].shape)
arg_dict['fc2_weight'][:] = np.random.normal(0, 0.2, arg_dict['fc2_weight'].shape)
arg_dict['fc3_weight'][:] = np.random.normal(0, 0.0001, arg_dict['fc3_weight'].shape)

grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
print grad_dict["fc3_bias"].shape
print arg_dict
executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
model_executor = FcExecutor(executor=executor,
                    data=arg_dict["data"],
                    data_grad=grad_dict["data"],
                    act=executor.outputs[0])

print("model_executor", model_executor)

lr = mx.lr_scheduler.FactorScheduler(step=10, factor=.9)

optimizer = mx.optimizer.SGD(
    learning_rate = 0.001,
    momentum = 0.9,
    wd = 0.005,
    lr_scheduler = lr,
    clip_gradient = 10)

optim_states_fc1 = optimizer.create_state(0, grad_dict['fc1_weight']) 
optim_states_fc2 = optimizer.create_state(1, grad_dict['fc2_weight']) 
optim_states_fc3 = optimizer.create_state(2, grad_dict['fc3_weight'])



while True:
    temp_shape = (1, 6400)
    executor = executor.reshape(allow_up_sizing=True, data=temp_shape)
    if render: 
      # env.render()
      cv2.imshow("mxnet_rl", env.render('rgb_array'))
      cv2.waitKey(1)
    # print "train"
    cur_x = prepro(observation)
    # print cur_x,prev_x
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # print x.shape
    xs.append(x) # observation

    data = mx.nd.array(x.reshape(1,Z))
    data.copyto(model_executor.data)
    # model_executor.executor.forward()
    executor.forward(is_train=True)

    theta = executor.outputs[0].asnumpy()
    aprob = sigmoid(theta)
    # print "alpha:", alpha


    # theta = model_executor.act
    # print "theta:", theta.asnumpy()
    # prob = Softmax(theta)
    # aprob = alpha[0][0]
    # print 'aprob: ', aprob
    action = 2 if np.random.uniform() < aprob else 3
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
    observation, reward, done, info = env.step(action)

    reward_sum += reward
    # print reward
    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    grad = []
    if done: # an episode finished
      episode_number += 1

      # stack together all inputs, hidden states, action gradients, and rewards for this episode
      # epx = np.vstack(xs)
      # eph1 = np.vstack(hs1)
      # eph2 = np.vstack(hs2)
      # print eph1.shape, eph2.shape
      # print len(dlogps), len(drs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      xs,hs1,hs2,dlogps,drs = [],[],[],[],[] # reset array memory

      # compute the discounted reward backwards through time
      discounted_epr = discount_rewards(epr)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= (np.std(discounted_epr) + 1e-20)

      epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
      
      # print 'epdlogp: ', epdlogp[0]
      # grad = [mx.nd.array(epdlogp[0])]
      # print grad[0].shape
      # out_grad[:] = mx.nd.array(epdlogp[0])
      # label = [y]
      # losGrad_theta = LogLossGrad(alpha, label)
      # print 'losGrad_theta: ', losGrad_theta

      # loss_sum = 0
      # for gradients in epdlogp:
      #   loss_sum += gradients
      # loss_mean = loss_sum / len(epdlogp) 
      print "epdlogp: ", len(epdlogp)
      # executor.reshape(allow_up_sizing=True, data=(len(epdlogp), 6400))
      temp_shape = (len(epdlogp), 6400)
      executor = executor.reshape(allow_up_sizing=True, data=temp_shape)
      print 'executor.outputs.shape: ',executor.outputs[0].shape
      out_grad = mx.nd.zeros(executor.outputs[0].shape, ctx=dev)
      print 'out_grad shape: ', out_grad.shape

      out_grad[:] = epdlogp.reshape(len(epdlogp),1) #losGrad_theta

      executor.backward([out_grad])
      print "bp success"
      optimizer.update(0, arg_dict['fc1_weight'], grad_dict['fc1_weight'], optim_states_fc1)
      optimizer.update(1, arg_dict['fc2_weight'], grad_dict['fc2_weight'], optim_states_fc2)
      optimizer.update(2, arg_dict['fc3_weight'], grad_dict['fc3_weight'], optim_states_fc3)
      

      # boring book-keeping
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
      # if episode_number % 100 == 0: 
      #   print 'Saving...'

      reward_sum = 0
      observation = env.reset() # reset env
      prev_x = None



    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
      print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


  
