from keras.models import Model
from keras.layers import Input, MaxPool2D

from more_keras import ConvBlock, SEBlock, ResidualBlock


class AutoEncoder(Model):
    def __init__(self, in_dim: int, in_features: int, n_features: int, blocks: int):
        x = inp = Input((in_dim, in_dim, in_features))

        x = ConvBlock(n_features)(x)

        for n in range(1, blocks+1):
            x = MaxPool2D()(x)
            x = ConvBlock((2**n) * n_features)(x)
            x = ConvBlock((2**n) * n_features)(x)
            x = SEBlock()(x)

        super().__init__(inp, x)


if __name__ == "__main__":
    model = AutoEncoder(96, 3, 32, 4)
    model.compile()
    model.summary()
