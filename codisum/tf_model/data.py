import json
import pickle
from collections import Counter

import numpy as np
from nltk.tokenize import RegexpTokenizer
from pygments import highlight
from pygments.lexers import JavaLexer
from pygments.formatters import RawTokenFormatter


def read_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def save_vocabulary(vocab, path):
    pickle.dump(vocab, open(path, 'wb'))


def load_vocabulary(path):
    return pickle.load(open(path, 'rb'))


def pygment_mul_line(java_lines):
    string = '\n'.join(java_lines)
    if string == '':
        return list(), dict()
    x = highlight(string, JavaLexer(), RawTokenFormatter())
    x = str(x, encoding='utf-8')
    tokenList = list()
    variableDict = dict()
    nameNum, attNum, clsNum, fucNum = 0, 0, 0, 0
    otherDict = dict()
    floatNum, numberNum, strNum = 0, 0, 0
    for y in x.splitlines():
        ys = y.split('\t')
        # print(ys)
        s = eval(ys[1])
        if s == '\n':
            tokenList.append('<nl>')
        elif s == 'NewBlock':
            tokenList.append('<nb>')
        elif s.isspace():
            lines = s.count('\n')
            for _ in range(lines):
                tokenList.append('<nl>')
        elif "Token.Literal.Number.Float" == ys[0]:
            if s not in otherDict:
                sT = 'FLOAT{}'.format(floatNum)
                otherDict[s] = sT
                floatNum += 1
            tokenList.append(otherDict[s])
        elif ys[0].startswith('Token.Literal.Number'):
            if s not in otherDict:
                sT = 'NUMBER{}'.format(numberNum)
                otherDict[s] = sT
                numberNum += 1
            tokenList.append(otherDict[s])
        elif ys[0].startswith('Token.Literal.String'):
            if s not in otherDict:
                sT = 'STRING{}'.format(strNum)
                otherDict[s] = sT
                strNum += 1
            tokenList.append(otherDict[s])
        elif "Token.Name.Namespace" == ys[0]:
            tokenList.append('NAMESPACE')
        elif "Token.Comment.Single" == ys[0]:
            tokenList.append('SINGLE')
            tokenList.append('<nl>')
        elif "Token.Comment.Multiline" == ys[0]:
            lines = s.count('\n')
            for _ in range(lines):
                tokenList.append('COMMENT')
                tokenList.append('<nl>')
            tokenList.append('COMMENT')
        elif 'Token.Name.Decorator' == ys[0]:
            tokenList.append('@')
            tokenList.append(s[1:].lower())
        elif 'Token.Name' == ys[0]:
            if s not in variableDict:
                sT = 'n{}'.format(nameNum)
                variableDict[s] = sT
                nameNum += 1
            tokenList.append(s)
        elif 'Token.Name.Attribute' == ys[0]:
            if s not in variableDict:
                sT = 'a{}'.format(attNum)
                variableDict[s] = sT
                attNum += 1
            tokenList.append(s)
        elif 'Token.Name.Class' == ys[0]:
            if s not in variableDict:
                sT = 'c{}'.format(clsNum)
                variableDict[s] = sT
                clsNum += 1
            tokenList.append(s)
        elif 'Token.Name.Function' == ys[0]:
            if s not in variableDict:
                sT = 'f{}'.format(fucNum)
                variableDict[s] = sT
                fucNum += 1
            tokenList.append(s)
        else:
            a = s.splitlines()
            for i in a:
                if i != '' and not i.isspace():
                    tokenList.append(i)
                tokenList.append('<nl>')
            tokenList.pop()
    return tokenList, variableDict


def encode_one_hot(int_data, vocab_size):
    one_hots = np.zeros([len(int_data), vocab_size])
    for i, value in enumerate(int_data):
        one_hots[i, int(value)] = 1
        if value == 0:
            break
    return one_hots


def process_text(text):
    lines = text.splitlines()
    java_lines = list()
    diff_marks = list()
    f = False
    for line in lines:
        if len(line) < 1:
            continue
        if line.startswith('diff --git'):
            if line.endswith('.java'):
                f = True
            else:
                f = False
            continue
        if not f:
            continue
        if line.startswith('+++') or line.startswith('---'):
            continue
        st = line[0]
        line = line[1:].strip()
        if st == '@':
            java_lines.append('NewBlock ' + line[line.find('@@') + 3:].strip())
            diff_marks.append(2)
        elif st == ' ':
            # 这个只能处理行首或行尾有标识的注释，且不准确
            if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                java_lines.append('COMMENT')
            else:
                java_lines.append(line)
            diff_marks.append(2)
        elif st == '-':
            if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                java_lines.append('COMMENT')
            else:
                java_lines.append(line)
            diff_marks.append(1)
        elif st == '+':
            if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                java_lines.append('COMMENT')
            else:
                java_lines.append(line)
            diff_marks.append(3)
    return java_lines, diff_marks


def data2model(lines, marks):
    toks, var = pygment_mul_line(lines)
    mi = 0
    tok_marks = []
    for i in toks:
        tok_marks.append(marks[mi])
        if i == '<nl>':
            mi += 1
            if mi >= len(marks):
                mi = len(marks) - 1
    return tok_marks, toks, var


def generator(data, batch_size, msg_vocab, shuffle=False):
    i = 0
    length, max_msg = data[3].shape
    while True:
        if i + batch_size > length:
            i = 0
            if shuffle:
                idx = np.arange(length)
                np.random.shuffle(idx)
                data = [d[idx, ...] for d in data]
        batch = [d[i: i+batch_size] for d in data]
        x = [batch[0], batch[1], batch[2], batch[3][:, :max_msg-1]]
        y = np.array([encode_one_hot(m, msg_vocab) for m in batch[3][:, 1:]])
        i += batch_size
        yield x, y


class SubWordTokenizer():
    """
    用于对标志符(下划线,字母,数字)进行分词得到一系列子词,一些分词实例:
    J2SE_CODE: [j, 2, se, code]
    RegexpTokenizer: [regexp, tokenizer]
    AST2NodePath: [ast, 2, node, path]
    """

    def __init__(self, lemmatize=False):
        self.tokenizer = RegexpTokenizer(pattern='[A-Za-z]+|\d+')
        self.alpha_tkn = RegexpTokenizer(pattern='[A-Z][a-z]*|[a-z]+')
        self.lemmatize = lemmatize
        if lemmatize:
            self.wnl = WordNetLemmatizer()

    def tokenize(self, text):
        tokens = list()
        _tokens = self.tokenizer.tokenize(text)
        for i in _tokens:
            if i.isdigit() or i.islower():
                tokens.append(i)
            elif i.isupper():
                tokens.append(i.lower())
            else:
                [tokens.append(j.lower()) for j in self.alpha_tkn.tokenize(i)]
        if self.lemmatize:
            tokens = [self.wnl.lemmatize(i, wordnet.VERB) for i in tokens]
        return tokens


class Vocabulary():
    def __init__(self):
        self.tok2id = {'<eos>': 0, '<sos>': 1, '<unkm>': 2}
        self.id2tok = ['<eos>', '<sos>', '<unkm>']
        self.eos = '<eos>'
        self.sos = '<sos>'
        self.unkm = '<unkm>'
        self.unkd = '<unkd>'
        self.msg_tok_num = 0
        self.ph_tok_num = 0
        self.idx = 3
        self.var_tkn = SubWordTokenizer()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.tok2id, self.id2tok, self.msg_tok_num,
                         self.ph_tok_num, self.idx], f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.tok2id, self.id2tok, self.msg_tok_num, \
                self.ph_tok_num, self.idx = pickle.load(f)

    def build_vocabulary(self, data, threshold=5):
        msg_counter = Counter()
        ph_set = set()
        code_set = set()
        for com in data:
            msg_counter.update(set(com['msg'].split()))
            for var, ph in com['var'].items():
                ph_set.add(ph)
                code_set.update(self.var_tkn.tokenize(var))
            for tok in com['diff'].split():
                if tok not in com['var']:
                    code_set.add(tok)
        for tok, num in msg_counter.items():
            if num >= threshold:
                self.tok2id[tok] = self.idx
                self.idx += 1
                self.id2tok.append(tok)
        self.msg_tok_num = self.idx
        # print(ph_set)
        # 为词汇表添加place holder相关词汇
        c, f, a, n = 15, 30, 50, 150    # 根据经验设置
        for i in range(c):
            self.tok2id['c{}'.format(i)] = self.idx
            self.idx += 1
            self.id2tok.append('c{}'.format(i))
        for i in range(f):
            self.tok2id['f{}'.format(i)] = self.idx
            self.idx += 1
            self.id2tok.append('f{}'.format(i))
        for i in range(a):
            self.tok2id['a{}'.format(i)] = self.idx
            self.idx += 1
            self.id2tok.append('a{}'.format(i))
        for i in range(n):
            self.tok2id['n{}'.format(i)] = self.idx
            self.idx += 1
            self.id2tok.append('n{}'.format(i))
        self.ph_tok_num = self.idx
        # 为词汇表添加其他词汇
        self.tok2id['<unkd>'] = self.idx
        self.idx += 1
        for tok in code_set:
            if tok not in self.tok2id:
                self.tok2id[tok] = self.idx
                self.idx += 1

    def encode_diff(self, mark, word, var, max_code, max_attr):
        d_mark = np.zeros([1, max_code], dtype=np.int)
        d_word = np.zeros([1, max_code], dtype=np.int)
        d_attr = np.zeros([1, max_code, max_attr], dtype=np.int)
        # 处理mark
        mark = mark[:max_code]
        d_mark[0, :len(mark)] = mark
        # 处理word和attr
        for j, t in enumerate(word):
            if j >= max_code:
                break
            # 处理attr
            if t in var:
                ats = self.var_tkn.tokenize(t)[:max_attr]
                for k, at in enumerate(ats):
                    idx = self.tok2id.get(at, self.tok2id[self.unkd])
                    d_attr[0, j, k] = idx
            # 处理word
            t = var.get(t, t)
            idx = self.tok2id.get(t, self.tok2id[self.unkd])
            d_word[0, j] = idx
        return d_mark, d_word, d_attr

    def data_trans(self, data, max_code, max_attr, max_msg):
        """ max_code: 输入diff的最大长度
            max_attr: 每个变量的子词的最大个数
            max_msg: msg的最大长度，输出会加sos，长度+1
        """
        length = len(data)
        d_mark = np.zeros([length, max_code], dtype=np.int)
        d_word = np.zeros([length, max_code], dtype=np.int)
        d_attr = np.zeros([length, max_code, max_attr], dtype=np.int)
        msg = np.zeros([length, max_msg + 1], dtype=np.int)
        for i, com in enumerate(data):
            # 处理msg
            msg[i, 0] = self.tok2id[self.sos]
            for j, t in enumerate(com['msg'].split()):
                if j >= max_msg:
                    break
                idx = self.tok2id.get(t, self.idx)
                if idx < self.msg_tok_num:
                    msg[i, j + 1] = idx
                else:
                    t = com['var'].get(t, self.unkm)
                    idx = self.tok2id.get(t, self.idx)
                    if idx < self.ph_tok_num:
                        msg[i, j + 1] = idx
                    else:
                        msg[i, j + 1] = self.tok2id[self.unkm]
            # 处理mark
            mark = com['mark'][:max_code]
            d_mark[i, :len(mark)] = mark
            # 处理word和attr
            for j, t in enumerate(com['diff'].split()):
                if j >= max_code:
                    break
                # 处理attr
                if t in com['var']:
                    ats = self.var_tkn.tokenize(t)[:max_attr]
                    for k, at in enumerate(ats):
                        idx = self.tok2id.get(at, self.tok2id[self.unkd])
                        d_attr[i, j, k] = idx
                # 处理word
                t = com['var'].get(t, t)
                idx = self.tok2id.get(t, self.tok2id[self.unkd])
                d_word[i, j] = idx
        return d_mark, d_word, d_attr, msg

    def decode(self, ids, var):
        var_r = {v: k for k, v in var.items()}
        toks = list()
        for i in ids:
            if i == 0:
                break
            else:
                if i >= self.ph_tok_num:
                    toks.append(self.unkm)
                tok = self.id2tok[i]
                toks.append(var_r.get(tok, tok))
        return " ".join(toks)

    def get_mask(self):
        mask = np.zeros([self.ph_tok_num, ], dtype=np.int)
        mask[:self.msg_tok_num] = 1
        return mask


def test_Vocabulary():
    vocab = Vocabulary()
    train = read_data('./dataset/train.json')
    vocab.build_vocabulary(train)
    # save_vocabulary(vocab, './dataset/vocab.pkl')
    # vocab = load_vocabulary('./dataset/vocab.pkl')
    vocab.save('./dataset/vocab.pkl')
    a, b, c, d = vocab.data_trans(train, 200, 5, 20)
    print(train[0])
    print(a[0])
    print(b[0])
    print(c[0])
    print(d[0])
    import torch
    a = torch.tensor(a)
    b = torch.tensor(b)
    c = torch.tensor(c)
    d = torch.tensor(d)
    print(vocab.msg_tok_num, vocab.ph_tok_num, vocab.idx)
    # print(vocab.tok2id)


if __name__ == '__main__':
    test_Vocabulary()
