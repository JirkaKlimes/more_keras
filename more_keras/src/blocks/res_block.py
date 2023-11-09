from keras.layers import Layer, Conv2D, BatchNormalization, Activation, Add


class ResidualBlock(Layer):
    def __init__(self, n_features: int, activation: str = 'silu', kernel_size: int = 3, padding: str = 'same', **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.conv1 = Conv2D(
            self.n_features, self.kernel_size, padding=self.padding)
        self.bn1 = BatchNormalization()
        self.act1 = Activation(self.activation)

        self.conv2 = Conv2D(
            self.n_features, self.kernel_size, padding=self.padding)
        self.bn2 = BatchNormalization()
        self.act2 = Activation(self.activation)
        return super().build(input_shape)

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.act2(x)
        return x
