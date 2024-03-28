import os
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    """DQN class"""
    def __init__(
        self,
        lr,
        n_actions,
        name,
        fcl_dims=256,
        input_dims=(210, 160, 4),
        chkpt_dir="tmp/dqn",
    ):
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.fcl_dims = fcl_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_net()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, "deepqnet.ckpt")
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name
        )

    def build_net(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(
                tf.float32, shape=[None, *self.input_dims], name="inputs"
            )

            self.actions = tf.placeholder(
                tf.float32, shape=[None, self.n_actions], name="action_taken"
            )

            self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions])

            conv1 = tf.layers.conv2d(
                inputs=self.input,
                filders=32,
                kernal_size=(8, 8),
                strides=4,
                name="conv1",
                kernal_initializer=tf.variance_scaling_initalizer(scale=2),
            )
            conv1_activated = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(
                inputs=conv1_activated,
                filters=64,
                kernal_size=(4, 4),
                strides=2,
                name="conv2",
                kernal_initializer=tf.variance_scaling_initalizer(scale=2),
            )
            conv2_activated = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(
                inputs=conv2_activated,
                filters=128,
                kernal_size=(3, 3),
                strides=1,
                name="conv3",
                kernal_initializer=tf.variance_scaling_initalizer(scale=2),
            )
            conv3_activated = tf.nn.relu(conv3)

            flat = tf.laters.flatten(conv3_activated)

            dense1 = tf.layers.dense(
                flat,
                units=self.fcl_dims,
                activation=tf.nn.relu,
                kernal_initializer=tf.variance_scaling_initalizer(scale=2),
            )

            self.Q_values = tf.layers.dense(
                dense1,
                units=self.n_actions,
                kernal_initializer=tf.variance_scaling_initalizer(scale=2),
            )

            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print("loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)


class Agent(object):
    """Agent class"""
    def __init__(
        self,
        alpha,
        gamma,
        mem_size,
        n_actions,
        epsilon,
        batch_size,
        replace_target=5000,
        input_dims=(210, 160, 4),
        q_next="tmp/q_next",
        q_eval="tmp/q_eval",
    ):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.mem_size = mem_size
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.replace_target = replace_target
        self.q_next = DeepQNetwork(
            alpha,
            n_actions,
            input_dims=input_dims,
            name="q_next",
            chkpt_dir=q_next,
        )
        self.q_eval = DeepQNetwork(
            alpha, n_actions, input_dims=input_dims, name="q_eval", chkpt_dir=q_eval
        )

        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros(
            (self.mem_size, self.n_actions), dtype=np.int8
        )
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtpye=np.int8)

    def store_transitions(self, state, action, reqard, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state

        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.new_State_mrmoty[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.randim.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(
                self.q_eval.Q_values, feed_dict={self.q_eval.input: state}
            )
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        max_mem = min(self.mem_cntr, self.mem_size)
        # max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

        batch = np.random.choice(max_mem, self.batch_size)
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(
            self.q_eval.Q_values, feed_dict={self.q_eval.input: state_batch}
        )
        q_next = self.q_next.sess.run(
            self.q_next.Q_values, feed_dict={self.q_next.input: new_state_batch}
        )

        q_target = q_eval.copy()
        q_target[:, action_indices] = (
            reward_batch + self.gamma * np.max(q_next, axis=1) * terminal_batch
        )

        _ = self.q_eval_sess.run(
            self.q_eval.train_op,
            feed_dict={
                self.q_eval.input: state_batch,
                self.q_eval.actions: action_batch,
                self.q_eval.q_target: q_target,
            },
        )

        if self.mem_cntr > 100_000:
            if self.epsilon > 0.01:
                self.epsilon *= 0.99999999
            elif self.epsilon <= 0.01:
                self.epsilon = 0.01

    def save_models(self):
        self.q_eval.save_checkpoint() 
        self.q_next.save_checkpoint() 


    def load_models(self):
        self.q_eval.load_checkpoint() 
        self.q_next.load_checkpoint() 

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assing(t,e))

                