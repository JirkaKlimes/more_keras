from keras.models import Model
from keras.layers import Input

from more_keras import ConvBlock


class AutoEncoder(Model):
    def __init__(self, in_dim: int, in_features: int, n_features: int, blocks: int):
        x = inp = Input((in_dim, in_dim, in_features))

        x = ConvBlock(32)(x)

        super().__init__(inp, x)


if __name__ == "__main__":
    model = AutoEncoder(96, 3, 32, 4)
    model.compile()
    model.summary()
