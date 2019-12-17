import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn
import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
cudnn.benchmark = True
if USE_CUDA:
    print("*---GPU可用,正在使用---*")
else: print("*---注意：GPU不可用---*")
#if torch.cuda.device_count()>1:
  #print("Use",torch.cuda.device_count(),"GPUs.")
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
#todo:zip_longest函数是已元素最多对象为基准,使用fillvalue的值来填充
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def translate_input2indexVar(l, voc):
    #todo:把输入batch string化为index， 同时padding
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)#64-bit integer
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def translate_output2indexVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    # todo:举例当batchsize=2，时，zeroPadding把[[146, 318, 3659, 1], [238, 155, 419, 1]] 两个回答合并为->padList=[(146, 238), (318, 155), (3659, 419), (1, 1)] padList【0】为所有回答的第一个词
    #todo：mask记录padVar[i][j]（第j个回答的第i个词）已经结束，为pad
    #todo：why input不记录pad??
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

#將 load.py 所整理好的training pairs，轉換成input, output Variable。
def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    #TODO：sort接受参数类型为函数，作为排序条件，此处用lambda函数x:=len(x[0].split(" ") 做排序参数->根据单词数reverse = True 降序， reverse = False 升序（默认）。
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # 將 input 和 output batch 做 padding 後，轉換成 Variable
    # 記錄 lengths 做 packed sequence 用
    # mask 是用來計算 loss
    inp, lengths = translate_input2indexVar(input_batch, voc)
    output, mask, max_target_len = translate_output2indexVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#############################################
# Training
#############################################

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_len 

#準備好 training data ，建好 training 所需要的 module 和 optimizer ，並重複的把 training batch 餵進 model 裡面做參數的更新。
#   main训练时调用：trainIters(args.train, reverse, n_iteration, learning_rate, batch_size,  n_layers, hidden_size, print_every, save_every, dropout)
def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)
    #todo:string转数字的字典，pairs为等待转换的对话

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    #todo:training_batches=随机抽取64组对话，交给batch2TrainData构成一组batch
    #TODO:没有采用epoch的模式，batch2TrainData负责將 load.py 所整理好的training pairs，轉換成input, output Variable。 总计循环n_iteration次，
    #TODO: 每次iteration调用batch2TrainData构造一个batch。每个batch为随机抽取64组对话，交给batch2TrainData构成一组batch。 因此此处有待改造
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
    # # model
    checkpoint = None
    print('Building encoder and decoder ...')

    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
   # if torch.cuda.device_count()>1:
       # encoder=nn.DataParallel(encoder)
       #decoder=nn.DataParallel(decoder)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            #print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            with open('log.txt','a') as f:
                import time
                template=' Iter: {:0>6d} process: {:.2f} avg_loss: {:.4f} time: {}\n'
                str=template.format(iteration, iteration/n_iteration*100,print_loss_avg,time.asctime(time.localtime(time.time())))
                f.write(str)
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
