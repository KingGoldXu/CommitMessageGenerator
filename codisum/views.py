import os
import pickle

import numpy as np
from django.contrib import messages
from django.shortcuts import render

from .forms import TextLangForm
from .tf_model.config import conf
from .tf_model.data import Vocabulary, data2model, process_text, read_data
from .tf_model.model import CoDiSumModel

vocab = Vocabulary()
vocab.load(os.path.join(conf.data_path, 'vocab.pkl'))
gen_mask = vocab.get_mask()
model, encoder, decoder = CoDiSumModel(conf.max_code, conf.max_msg, conf.attr_num,
                                       vocab.idx, vocab.ph_tok_num, conf.mark_embed,
                                       conf.word_embed, conf.hid_dim, conf.attn_num,
                                       0., gen_mask)
model.load_weights(os.path.join(conf.model_path, 'codisum.h5'))
# 开业大吉
text = """diff --git a/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java b/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
index 9fba460..0fd1746 100644
--- a/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
+++ b/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
@@ -121,9 +121,12 @@ public final class CacheTest {
     assertCached(false, 207);
     assertCached(true, 300);
     assertCached(true, 301);
-    for (int i = 302; i <= 307; ++i) {
-      assertCached(false, i);
-    }
+    assertCached(true, 302);
+    assertCached(false, 303);
+    assertCached(false, 304);
+    assertCached(false, 305);
+    assertCached(false, 306);
+    assertCached(true, 307);
     assertCached(true, 308);
     for (int i = 400; i <= 406; ++i) {
       assertCached(false, i);"""
a, b = process_text(text)


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


def predict_multi_sentence(encoder, decoder, vocab, lines, marks):
    mark, word, var = data2model(lines, marks)
    d = vocab.encode_diff(mark, word, var, conf.max_code,
                          conf.attr_num, conf.max_msg)
    en_r = encoder.predict(list(d))
    cur_token = np.zeros((1, 1))
    cur_token[0, 0] = 1
    beams = [[cur_token, d[1]] + en_r + [[], 0]]
    for _ in range(conf.max_msg):
        beams = beam_decode_step(decoder, beams, conf.beam_size)
    messages = dict()
    for i, beam in enumerate(beams):
        gen = vocab.decode(beam[-2], var)
        messages[gen] = ("%.2f" % beam[-1])
    return messages


# 必须在启动前调用一次predict，不然会报错
predict_multi_sentence(encoder, decoder, vocab, a, b)


def generated(request):
    if request.method == "POST":
        form = TextLangForm(request.POST)
        if form.is_valid():
            Text = form.cleaned_data['text']
            a, b = process_text(Text)
            if a:
                res = predict_multi_sentence(encoder, decoder, vocab, a, b)
                return render(request, 'codisum/form.html', {'text': Text, 'res_dict': res})
            else:
                messages.success(
                    request, "input does not contain valid information")
    return render(request, 'codisum/form.html', {'text': "", 'res_dict': dict()})


def about(request):
    return render(request, 'codisum/about.html')


if __name__ == '__main__':
    text = """diff --git a/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java b/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
index 9fba460..0fd1746 100644
--- a/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
+++ b/okhttp-tests/src/test/java/com/squareup/okhttp/CacheTest.java
@@ -121,9 +121,12 @@ public final class CacheTest {
     assertCached(false, 207);
     assertCached(true, 300);
     assertCached(true, 301);
-    for (int i = 302; i <= 307; ++i) {
-      assertCached(false, i);
-    }
+    assertCached(true, 302);
+    assertCached(false, 303);
+    assertCached(false, 304);
+    assertCached(false, 305);
+    assertCached(false, 306);
+    assertCached(true, 307);
     assertCached(true, 308);
     for (int i = 400; i <= 406; ++i) {
       assertCached(false, i);"""
    a, b = process_text(text)
    res = predict_multi_sentence(encoder, decoder, vocab, a, b)
    print(res)
