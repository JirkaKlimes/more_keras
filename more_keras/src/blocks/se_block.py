from keras.layers import Layer, Conv2D, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Reshape


class SEBlock(Layer):
    def __init__(self, activation: str = 'silu', **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, input_shape):
        features = input_shape[-1]
        self.squeeze = GlobalAveragePooling2D()
        self.excite1 = Dense(features // 4, activation=self.activation)
        self.excite2 = Dense(features, activation='sigmoid')

        return super().build(input_shape)

    def call(self, x):
        squeeze = self.squeeze(x)
        excite = self.excite1(squeeze)
        excite = self.excite2(excite)
        excite = Reshape((1, 1, -1))(excite)
        x *= excite
        return x
