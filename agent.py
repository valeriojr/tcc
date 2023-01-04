import numpy
import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import activations


class Actor:
    def __init__(self, state_shape, actions, *args, **kwargs):
        super().__init__(*args, **kwargs)

        state = keras.Input(shape=state_shape)
        # distance = keras.Input(shape=(1,))

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(state)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        # d = layers.Reshape((1,))(distance)
        # x = layers.Concatenate()([x, d])
        # x = layers.Dense(units=64, activation='relu')(x)
        # x = layers.Dense(units=32, activation='relu')(x)
        # x = layers.Dense(units=16, activation='relu')(x)
        # x = layers.Dense(units=8, activation='relu')(x)

        actions = [layers.Dense(units=5, activation='softmax')(x) for i in range(actions)]

        # self.model = keras.Model(inputs=[state, distance], outputs=actions)
        self.model = keras.Model(inputs=[state], outputs=actions)
        self.model.compile()

    @staticmethod
    def get_action_loss(prob, index, td):
        dist = distributions.Categorical(probs=prob, dtype=tf.float32)

        return -dist.log_prob(index) * td

    def get_loss(self, probs, action, td):
        return tf.reduce_sum([
            self.get_action_loss(probs[i], action[i], td) for i in range(len(self.model.outputs))
        ])

    def __call__(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def update(self, tape, loss):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class Critic():
    def __init__(self, state_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

        state = keras.Input(shape=state_shape)
        # distance = keras.Input(shape=(1,))

        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(state)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        # d = layers.Reshape((1,))(distance)
        # x = layers.Concatenate()([x, d])
        x = layers.Dense(units=64, activation='relu')(x)
        x = layers.Dense(units=32, activation='relu')(x)
        x = layers.Dense(units=16, activation='relu')(x)
        x = layers.Dense(units=8, activation='relu')(x)

        value = layers.Dense(units=1, activation=activations.leaky_relu)(x)

        # self.model = keras.Model(inputs=[state, distance], outputs=[value])
        self.model = keras.Model(inputs=[state], outputs=[value])
        self.model.compile()

    def __call__(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def update(self, tape, loss):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class Agent:
    def __init__(self, state_shape, actions, learning_rate, discount_factor, saved_actor='', saved_critic=''):
        self.actions = actions
        self.actor = Actor(state_shape, actions)
        self.critic = Critic(state_shape)

        try:
            self.actor.model = keras.models.load_model(saved_actor)
            self.critic.model = keras.models.load_model(saved_critic)
        except OSError:
            pass

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def act(self, state):
        return self.actor(state)

    def criticize(self, value, reward, next_state):
        next_value = self.critic(next_state)

        return reward + (self.discount_factor * next_value) - value

    def get_actor_loss(self, action, td):
        return self.actor.get_loss(action, self.sample_actions(action), td)

    def update(self, actor_tape, actor_loss, critic_tape, critic_loss):
        self.actor.update(actor_tape, actor_loss)
        self.critic.update(critic_tape, critic_loss)

    def sample_actions(self, action):
        prob = numpy.array(action)
        dist = distributions.Categorical(probs=prob)

        return dist.sample().numpy()
