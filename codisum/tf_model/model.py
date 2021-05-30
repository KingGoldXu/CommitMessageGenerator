import numpy as np
from keras import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import (GRU, Bidirectional, Concatenate, Dense, Dropout,
                          Embedding, Input, Lambda, Reshape, TimeDistributed)


class ComputeAttention(Layer):
    # compute the attention between encoder hidden state and decoder hidden state
    def __init__(self, units, **kwargs):
        super(ComputeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0

    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        # w1
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # w2
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # nu
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        # Be sure to call this somewhere!
        super(ComputeAttention, self).build(input_shape)

    def call(self, x, mask=None):
        # x[0] is encoder hidden state, x[1] is decoder hidden state.
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.int_shape(de_seq)[-2]

        use_mask = False
        if len(x) == 3:
            mask = x[2]
            m_en = K.cast(mask, K.floatx())
            use_mask = True
        if len(x) == 2 and mask is not None:
            m_en = K.cast(mask[0], K.floatx())
            use_mask = True

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(
            att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(
            att_en, shape=(-1, input_de_times * self.input_en_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        if use_mask:
            m_en = K.repeat(m_en, input_de_times)
            m_en = K.reshape(m_en, shape=(-1, 1))
            m_en = m_en - 1
            m_en = m_en * 1000000
            mu = mu + m_en

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        return K.softmax(mu)

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][1], input_shape[0][1]


class Masked(Layer):
    def __init__(self, return_mask=False, **kwargs):
        self.supports_masking = True
        self.return_mask = return_mask
        super(Masked, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # Be sure to call this somewhere!
        super(Masked, self).build(input_shape)

    def call(self, x, mask=None):
        output = x
        if mask is not None:
            # remove padding values
            m = K.cast(mask, K.floatx())
            output = x * K.expand_dims(m, -1)
        if self.return_mask:
            return [output, mask]
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_mask:
            return [output_shape, output_shape[:-1]]
        return output_shape


class MaskedSoftmax(Layer):
    def __init__(self, mask, **kwargs):
        self.mask = mask
        self.supports_masking = True
        super(MaskedSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mask = K.constant(self.mask, K.floatx())
        m_en = mask - 1
        m_en = m_en * 1000000
        m_en = K.expand_dims(m_en, 0)
        inputs = inputs + m_en
        inputs = K.softmax(inputs)
        return inputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape


class AttentionCopy(Layer):
    def __init__(self, size, **kwargs):
        self.size = size
        super(AttentionCopy, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionCopy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs is a list, x[0] is encoder word input, x[1] is attention alpha
        in_one_hot = K.one_hot(K.cast(inputs[0], 'int32'), self.size)
        output = K.batch_dot(inputs[1], in_one_hot)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        y = list(input_shape[1])
        y[-1] = self.size
        return tuple(y)


class CombineGenCopy(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(CombineGenCopy, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CombineGenCopy, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs[0] is p_gen, inputs[1] is gen_prob, inputs[2] is copy_prob
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def CoDiSumModel(len_en, len_de, attr_num, embed_vocab_size, decode_vocab_size,
                 m_embed_dim, w_embed_dim, hid_size, att_num, drop_rate,
                 gen_mask):
    a = np.random.random([m_embed_dim])
    b = np.zeros([m_embed_dim])
    weight = np.array([[b, -a, b, a]])
    mark_embed_layer = Embedding(4, m_embed_dim, mask_zero=True,
                                 weights=weight, trainable=True)
    word_embed_layer = Embedding(embed_vocab_size, w_embed_dim, mask_zero=True)
    bi_rnn_layer1 = Bidirectional(GRU(hid_size, return_sequences=True,
                                      return_state=True))
    bi_rnn_layer2 = Bidirectional(GRU(hid_size, return_sequences=True,
                                      return_state=True))
    bi_rnn_layer3 = Bidirectional(GRU(hid_size, return_sequences=True,
                                      return_state=True))
    bi_rnn_layer4 = Bidirectional(GRU(hid_size, return_sequences=True))
    bi_rnn_layer5 = Bidirectional(GRU(hid_size, return_sequences=True))
    bi_rnn_layer6 = Bidirectional(GRU(hid_size))
    rnn_layer1 = GRU(hid_size * 2, return_sequences=True, return_state=True)
    rnn_layer2 = GRU(hid_size * 2, return_sequences=True, return_state=True)
    rnn_layer3 = GRU(hid_size * 2, return_sequences=True, return_state=True)
    compute_alpha = ComputeAttention(att_num)
    p_gen_dense_layer = Dense(1, activation='sigmoid')
    gen_dense_layer = Dense(decode_vocab_size)
    dropout = Dropout(drop_rate)

    m_encoder_in = Input(shape=(len_en,), dtype=K.floatx())
    w_encoder_in = Input(shape=(len_en,), dtype=K.floatx())
    a_encoder_in = Input(shape=(len_en, attr_num), dtype=K.floatx())
    a_reshape_in = Lambda(lambda x: K.reshape(x, (-1, attr_num)))(a_encoder_in)
    m_embed_en = mark_embed_layer(m_encoder_in)
    w_embed_en = word_embed_layer(w_encoder_in)
    a_embed_en = word_embed_layer(a_reshape_in)
    embed_en = Concatenate()([m_embed_en, w_embed_en])
    rnn_h1, state_f1, state_b1 = bi_rnn_layer1(embed_en)
    rnn_h2, state_f2, state_b2 = bi_rnn_layer2(dropout(rnn_h1))
    rnn_h3, state_f3, state_b3 = bi_rnn_layer3(dropout(rnn_h2))
    a_rnn_h1 = bi_rnn_layer4(a_embed_en)
    a_rnn_h2 = bi_rnn_layer5(dropout(a_rnn_h1))
    a_rnn_h3 = bi_rnn_layer6(dropout(a_rnn_h2))
    a_rnn_h3 = Lambda(lambda x: K.reshape(
        x, (-1, len_en, hid_size * 2)))(dropout(a_rnn_h3))
    state1 = Concatenate()([state_f1, state_b1])
    state2 = Concatenate()([state_f2, state_b2])
    state3 = Concatenate()([state_f3, state_b3])
    rnn_h3 = dropout(rnn_h3)
    masked_rnn_h3, mask = Masked(return_mask=True)(rnn_h3)
    masked_rnn_h3 = Concatenate()([masked_rnn_h3, a_rnn_h3])
    encoder = Model(inputs=[m_encoder_in, w_encoder_in, a_encoder_in],
                    outputs=[masked_rnn_h3, mask, m_embed_en, state1,
                             state2, state3])
    # print(encoder.summary())

    token_in = Input(shape=(1,), dtype=K.floatx())
    state1_p = Input(shape=(hid_size * 2,), dtype=K.floatx())
    state2_p = Input(shape=(hid_size * 2,), dtype=K.floatx())
    state3_p = Input(shape=(hid_size * 2,), dtype=K.floatx())
    hid_state_en = Input(shape=(len_en, hid_size * 4), dtype=K.floatx())
    mask_en = Input(shape=(len_en,), dtype='bool')
    mark_en = Input(shape=(len_en, m_embed_dim))
    embed_de = word_embed_layer(token_in)
    rnn_h4_p, state_out1 = rnn_layer1(embed_de, initial_state=state1_p)
    rnn_h5_p, state_out2 = rnn_layer2(rnn_h4_p, initial_state=state2_p)
    rnn_h6_p, state_out3 = rnn_layer3(rnn_h5_p, initial_state=state3_p)
    alpha = compute_alpha([hid_state_en, rnn_h6_p, mask_en])
    att_cont = Lambda(lambda x: K.batch_dot(
        x[0], x[1], axes=(2, 1)))([alpha, hid_state_en])
    att_mark = Lambda(lambda x: K.batch_dot(
        x[0], x[1], axes=(2, 1)))([alpha, mark_en])
    p_gen_source = Concatenate()([rnn_h6_p, att_cont, embed_de])
    p_gen = p_gen_dense_layer(p_gen_source)
    att_out = Concatenate()([rnn_h6_p, att_cont, att_mark])
    gen_prob = TimeDistributed(gen_dense_layer)(att_out)
    gen_prob = MaskedSoftmax(gen_mask)(gen_prob)
    copy_prob = AttentionCopy(decode_vocab_size)([w_encoder_in, alpha])
    next_token = CombineGenCopy()([p_gen, gen_prob, copy_prob])
    decoder = Model([token_in, w_encoder_in, hid_state_en, mask_en, mark_en,
                     state1_p, state2_p, state3_p],
                    [next_token, p_gen, alpha, state_out1, state_out2,
                     state_out3])
    # print(decoder.summary())

    decoder_in = Input(shape=(len_de,), dtype=K.floatx())
    embed_de = word_embed_layer(decoder_in)
    rnn_h4, _ = rnn_layer1(embed_de, initial_state=state1)
    rnn_h5, _ = rnn_layer2(dropout(rnn_h4), initial_state=state2)
    rnn_h6, _ = rnn_layer3(dropout(rnn_h5), initial_state=state3)
    rnn_h6 = dropout(rnn_h6)
    alpha = compute_alpha([masked_rnn_h3, rnn_h6, mask])
    att_cont = Lambda(lambda x: K.batch_dot(
        x[0], x[1], axes=(2, 1)))([alpha, masked_rnn_h3])
    att_mark = Lambda(lambda x: K.batch_dot(
        x[0], x[1], axes=(2, 1)))([alpha, m_embed_en])
    att_cont = dropout(att_cont)
    att_mark = dropout(att_mark)
    p_gen_source = Concatenate()([rnn_h6, att_cont, embed_de])
    p_gen = p_gen_dense_layer(p_gen_source)
    att_out = Concatenate()([rnn_h6, att_cont, att_mark])
    gen_prob = TimeDistributed(gen_dense_layer)(att_out)
    gen_prob = MaskedSoftmax(gen_mask)(gen_prob)
    copy_prob = AttentionCopy(decode_vocab_size)([w_encoder_in, alpha])
    output = CombineGenCopy()([p_gen, gen_prob, copy_prob])
    model = Model(inputs=[m_encoder_in, w_encoder_in,
                          a_encoder_in, decoder_in], outputs=output)
    # print(model.summary())
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model, encoder, decoder
