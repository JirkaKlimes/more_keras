from keras.layers import Layer, Conv2D, BatchNormalization, Activation


class ConvBlock(Layer):
    def __init__(self, n_features: int, activation: str, kernel_size: int = 3, padding: str = 'same', **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.activation = activation
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.conv = Conv2D(
            self.n_features, self.kernel_size, padding=self.padding)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)
        return super().build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
