import torch
import re
import os
import unicodedata

from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}#word转索引，字典，word字符串做查询键值
        self.word2count = {}#word转频次计数器，似乎没卵用
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#TODO: Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
#不过好像这是机器翻译用的，此处没有用上
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    #TODO:Python file.readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。如果碰到结束符 EOF 则返回空字符串。
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    #TODO：Python string.strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    pairs = [[x, next(it)] for x in it]
    #TODO:pairs.__len__()=225156,简单的把第i句和i+1句(i为奇数)组合为一组对话
    voc = Voc(corpus_name)
    #TODO:voc 代表词汇表，包含数字index和字符串word之间的相互转换关系，（查找表），方便将word变成onehot向量；pairs则代表对话中的问答对
    #TODO:此时voc只是生成了，但还没有处理pair的词汇
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name):
    #TODO:如果corpus_name（数据集名字）对应的数据不在save文件夹则调用本函数生成

    voc, pairs = readVocs(corpus, corpus_name)
    print("读取到 {!s} sentence pairs".format(len(pairs)))#'!s'在表达式上调用str（），'!r'调用表达式上的repr（），'!a'调用表达式上的ascii（）
    pairs = filterPairs(pairs)
    # TODO:根据config.py文件定义的MAX_LENGTH，句子长度（单词数字）超过MAX_LENGTH的对话将被删除
    print("删减后剩余 {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    #TODO: corpus=../data/movie_subtitles.txt
    #TODO:  corpus_name（数据集名字）= movie_subtitles
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
        print("载入对话 {!s} sentence pairs".format(len(pairs)))
        print("载入单词 {!s} words".format(voc.n_words))
        #TODO: save_dir 在config.py 定义，为./,所以voc=./training_data/movie_subtitles/voc.tar
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(corpus, corpus_name)
    return voc, pairs
