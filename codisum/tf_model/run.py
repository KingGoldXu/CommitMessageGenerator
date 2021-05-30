
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from config import conf
from data import Vocabulary, generator, read_data
from model import CoDiSumModel


def train_model():
    np.random.seed(conf.seed)
    train_data = read_data(os.path.join(conf.data_path, 'train.json'))
    valid_data = read_data(os.path.join(conf.data_path, 'valid.json'))
    vocab = Vocabulary()
    vocab.build_vocabulary(train_data)
    vocab.save(os.path.join(conf.data_path, 'vocab.pkl'))
    gen_mask = vocab.get_mask()
    train = vocab.data_trans(train_data, conf.max_code, conf.attr_num,
                             conf.max_msg)
    valid = vocab.data_trans(valid_data, conf.max_code, conf.attr_num,
                             conf.max_msg)
    model_path = os.path.join(conf.model_path, 'codisum.h5')
    model, _, _ = CoDiSumModel(conf.max_code, conf.max_msg, conf.attr_num,
                               vocab.idx, vocab.ph_tok_num, conf.mark_embed,
                               conf.word_embed, conf.hid_dim, conf.attn_num,
                               conf.drop_rate, gen_mask)
    model.compile('rmsprop', 'categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', patience=conf.patience)
    cp = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                         save_best_only=True)
    model.fit_generator(generator(train, conf.batch_size, vocab.ph_tok_num, True),
                        len(train[0]) // conf.batch_size, conf.epochs,
                        callbacks=[es, cp],
                        validation_data=generator(valid, conf.batch_size,
                                                  vocab.ph_tok_num),
                        validation_steps=len(valid[0]) // conf.batch_size)


def predict():
    test_data = read_data(os.path.join(conf.data_path, 'test.json'))
    vocab = Vocabulary()
    vocab.load(os.path.join(conf.data_path, 'vocab.pkl'))
    gen_mask = vocab.get_mask()
    model, encoder, decoder = CoDiSumModel(conf.max_code, conf.max_msg, conf.attr_num,
                                           vocab.idx, vocab.ph_tok_num, conf.mark_embed,
                                           conf.word_embed, conf.hid_dim, conf.attn_num,
                                           0., gen_mask)
    model_path = os.path.join(conf.model_path, 'codisum.h5')
    model.load_weights(model_path)
    for com in test_data:
        mark, word, attr, msg = vocab.data_trans([com], conf.max_code,
                                                 conf.attr_num, conf.max_msg)
        masked_rnn_h3, mask, m_embed_en, state1, state2, state3 = \
            encoder.predict([mark, word, attr])
        cur_token = np.zeros((1, 1))
        cur_token[0, 0] = 1
        results = []
        predict_next_token(decoder, cur_token, word, masked_rnn_h3, mask,
                           m_embed_en, state1, state2, state3, 0, 0.0,
                           results, [], [], [], 20, 2)
        results = sorted(results, key=lambda x: x[0], reverse=True)
        de_seq = results[0][1]
        gen = vocab.decode(de_seq, com['var'])
        print('Gold message: {}'.format(com['msg']))
        print('Pred message: {}'.format(gen))
        print('--------------------------------')


def predict_multi_sentence():
    test_data = read_data(os.path.join(conf.data_path, 'test.json'))
    vocab = Vocabulary()
    vocab.load(os.path.join(conf.data_path, 'vocab.pkl'))
    gen_mask = vocab.get_mask()
    model, encoder, decoder = CoDiSumModel(conf.max_code, conf.max_msg, conf.attr_num,
                                           vocab.idx, vocab.ph_tok_num, conf.mark_embed,
                                           conf.word_embed, conf.hid_dim, conf.attn_num,
                                           0., gen_mask)
    model_path = os.path.join(conf.model_path, 'codisum.h5')
    model.load_weights(model_path)
    for com in test_data:
        d = vocab.data_trans([com], conf.max_code, conf.attr_num, conf.max_msg)
        en_r = encoder.predict(list(d)[:3])
        cur_token = np.zeros((1, 1))
        cur_token[0, 0] = 1
        beams = [[cur_token, d[1]] + en_r + [[], 0]]
        for _ in range(conf.max_msg):
            beams = beam_decode_step(decoder, beams, conf.beam_size)
        print('Gold message: {}'.format(com['msg']))
        for i, beam in enumerate(beams):
            print('Res {} with score {}'.format(i + 1, beam[-1]))
            gen = vocab.decode(beam[-2], com['var'])
            print('Pred message: {}'.format(gen))
        print('--------------------------------')


def beam_decode_step(decoder, inputs, beam_size):
    """ inputs = [[cur_token, word_in, encoder_in, mask, m_embed, state1, state2, state3, pred_seq, score],]
        这个实现不考虑提前结束的token
    """
    tmp = []
    for one in inputs:
        if one[8] and one[8][-1] == 0:
            tmp.append(one)
            continue
        prs, pgen, alpha, st1, st2, st3 = decoder.predict(one[:8])
        prs = [(i, v) for i, v in enumerate(prs[0, 0, :])]
        prs = sorted(prs, key=lambda x: x[1], reverse=True)
        for idx, value in prs[:beam_size]:
            pred_seq = one[8] + [idx]
            score = one[9] + np.log(value)
            token = np.zeros((1, 1))
            token[0, 0] = idx
            tmp.append([token] + one[1:5] + [st1, st2, st3, pred_seq, score])
    tmp = sorted(tmp, key=lambda x: x[-1], reverse=True)
    return tmp[:beam_size]


def predict_next_token(decoder, cur_token, word_in, encoder_in, mask, m_embed, state1, state2, state3, cur_depth,
                       joint_prs, res, tags, alphas, pgens, max_len, beam_size):
    cur_depth += 1
    prs, pgen, alpha, st1, st2, st3 = decoder.predict([cur_token, word_in, encoder_in, mask, m_embed,
                                                       state1, state2, state3])
    prs = prs[0, 0, :]
    alpha = alpha[0, 0, :]
    pgen = pgen[0, 0, 0]
    # i for index, v for softmax value in i
    prs = [(i, v) for i, v in enumerate(prs)]
    prs = sorted(prs, key=lambda x: x[1], reverse=True)
    # if cur_depth == 2:                      # TODO
    #     beam_size = 2
    if cur_depth == 4:
        beam_size = 1
    for p in prs[:beam_size]:
        # end of sentence
        if p[0] == 0:
            res.append(
                ((joint_prs + np.log(p[1]))/cur_depth, tags, alphas, pgens))
            break
        # max decode len
        if cur_depth == max_len - 1:
            res.append(((joint_prs + np.log(p[1]))/cur_depth,
                        tags[:] + [p[0]], alphas + [alpha], pgens + [pgen]))
            break
        # generation continue
        token = np.zeros((1, 1))
        token[0, 0] = p[0]
        predict_next_token(decoder, token, word_in, encoder_in, mask, m_embed, st1, st2, st3, cur_depth,
                           joint_prs +
                           np.log(p[1]), res, tags[:] +
                           [p[0]], alphas + [alpha], pgens + [pgen],
                           max_len, beam_size)


def plot_models():
    vocab = Vocabulary()
    vocab.load(os.path.join(conf.data_path, 'vocab.pkl'))
    gen_mask = vocab.get_mask()
    model, encoder, decoder = CoDiSumModel(conf.max_code, conf.max_msg, conf.attr_num,
                                           vocab.idx, vocab.ph_tok_num, conf.mark_embed,
                                           conf.word_embed, conf.hid_dim, conf.attn_num,
                                           0., gen_mask)
    # plot_model(model, '/home/kingxu/CoDiSum_model.png')
    # plot_model(encoder, '/home/kingxu/CoDiSum_encoder.png')
    # plot_model(decoder, '/home/kingxu/CoDiSum_decoder.png')
    model.summary()


if __name__ == '__main__':
    # train_model()
    # predict()
    # predict_multi_sentence()
    plot_models()
