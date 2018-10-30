import argparse
import datetime
import os
import pickle

from tqdm import tqdm

from TextCNN import TextCNN
from expert_features import get_expert_features
from utils import get_word_vocab, read_data, ngram_id_x, get_words, char_id_x, prep_train_test, \
    get_ngramed_id_x, pad_seq_in_word, pad_seq, batch_iter
import tensorflow as tf
import numpy as np

CHAR = 1
WORD = 2
CHAR_AND_WORD = 3
CHARWORD_AND_WORD = 4
CHARWORD_AND_WORD_AND_CHAR = 5

data_size = 1000
add_expert_feature = 1
emb_mode = CHARWORD_AND_WORD_AND_CHAR

import sys

if len(sys.argv) > 2:

    parser = argparse.ArgumentParser(description="Train URLNet model")

    # data args
    default_max_len_words = 200
    parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words,
                        metavar="MLW",
                        help="maximum length of url in words (default: {})".format(
                            default_max_len_words))
    default_max_len_chars = 200
    parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars,
                        metavar="MLC",
                        help="maximum length of url in characters (default: {})".format(
                            default_max_len_chars))
    default_max_len_subwords = 20
    parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords,
                        metavar="MLSW",
                        help="maxium length of word in subwords/ characters (default: {})".format(
                            default_max_len_subwords))
    default_min_word_freq = 1
    parser.add_argument('--data.min_word_freq', type=int, default=default_min_word_freq,
                        metavar="MWF",
                        help="minimum frequency of word in training population to build vocabulary (default: {})".format(
                            default_min_word_freq))
    default_dev_pct = 0.1
    parser.add_argument('--data.dev_pct', type=float, default=default_dev_pct, metavar="DEVPCT",
                        help="percentage of training set used for dev (default: {})".format(
                            default_dev_pct))
    parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR",
                        help="location of data file")
    default_delimit_mode = 1
    parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode,
                        metavar="DLMODE",
                        help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(
                            default_delimit_mode))

    # model args
    default_emb_dim = 32
    parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
                        help="embedding dimension size (default: {})".format(default_emb_dim))
    default_filter_sizes = "3,4,5,6"
    parser.add_argument('--model.filter_sizes', type=str, default=default_filter_sizes,
                        metavar="FILTERSIZES",
                        help="filter sizes of the convolution layer (default: {})".format(
                            default_filter_sizes))
    default_emb_mode = 1
    parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
                        help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(
                            default_emb_mode))

    # train args
    default_nb_epochs = 5
    parser.add_argument('--train.nb_epochs', type=int, default=default_nb_epochs, metavar="NEPOCHS",
                        help="number of training epochs (default: {})".format(default_nb_epochs))
    default_batch_size = 128
    parser.add_argument('--train.batch_size', type=int, default=default_batch_size,
                        metavar="BATCHSIZE",
                        help="Size of each training batch (default: {})".format(default_batch_size))

    parser.add_argument('--train.add_expert_feature', type=int, default=0,
                        metavar="ADDEXPERTREATURE",
                        help="add expert feature (default: {})".format(default_batch_size))

    parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA",
                        help="l2 lambda for regularization (default: 0.0)")
    default_lr = 0.001
    parser.add_argument('--train.lr', type=float, default=default_lr, metavar="LR",
                        help="learning rate for optimizer (default: {})".format(default_lr))

    # log args
    parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR",
                        help="directory of the output model")
    parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY",
                        help="print training result every this number of steps (default: 50)")
    parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY",
                        help="evaluate the model every this number of steps (default: 500)")
    parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY",
                        help="save a model every this number of steps (default: 500)")

    FLAGS = vars(parser.parse_args())
else:
    base_dir = 'runs/%d_emb%d_dlm1_32dim_minwf1_1conv3456_5ep_expert%d' % (data_size, emb_mode, add_expert_feature)
    FLAGS = {
        'log.checkpoint_dir': '%s/checkpoints/' % base_dir,
        'data.char_dict_dir': '%s/chars_dict.p' % base_dir,
        'model.emb_mode': emb_mode,
        'data.delimit_mode': 1,
        'data.max_len_words': 200,
        'test.batch_size': 10,
        'data.subword_dict_dir': '%s/subwords_dict.p' % base_dir,
        'data.data_dir': './data/test_%s.txt' % data_size,
        'data.word_dict_dir': '%s/words_dict.p' % base_dir,
        'log.output_dir': base_dir,
        'data.max_len_subwords': 20, 'data.max_len_chars': 200, 'model.emb_dim': 32,
        'data.min_word_freq': 1,
        'data.dev_pct': 0.1,
        'train.add_expert_feature': add_expert_feature,
        'train.l2_reg_lambda': 0.0,
        'train.lr': 0.001,
        'model.filter_sizes': "3,4,5,6",
        'train.batch_size': 10,
        'train.nb_epochs': 5,
        'log.print_every': 50,
        'log.eval_every': 50,
        'log.checkpoint_every': 50
    }

urls, labels = read_data(FLAGS["data.data_dir"])

high_freq_words = None
if FLAGS["data.min_word_freq"] > 0:
    x1, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"],
                                           FLAGS["data.min_word_freq"])
    high_freq_words = sorted(list(word_reverse_dict.values()))
    print("Number of words with freq >={}: {}".format(FLAGS["data.min_word_freq"],
                                                      len(high_freq_words)))

expert_features_x = get_expert_features(urls)
x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x,
                                                                FLAGS["data.max_len_subwords"],
                                                                high_freq_words)

chars_dict = ngrams_dict
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

pos_x = []
neg_x = []
for i in range(len(labels)):
    label = labels[i]
    if label == 1:
        pos_x.append(i)
    else:
        neg_x.append(i)
print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x)
neg_x = np.array(neg_x)

x_train_idx, y_train_idx, x_test_idx, y_test_idx = prep_train_test(pos_x, neg_x,
                                                                   FLAGS["data.dev_pct"])

x_train_wordchar_id = get_ngramed_id_x(x_train_idx, ngramed_id_x)
x_test_wordchar_id = get_ngramed_id_x(x_test_idx, ngramed_id_x)

x_train_word_id = get_ngramed_id_x(x_train_idx, worded_id_x)
x_test_word_id = get_ngramed_id_x(x_test_idx, worded_id_x)

x_train_char_id = get_ngramed_id_x(x_train_idx, chared_id_x)
x_test_char_id = get_ngramed_id_x(x_test_idx, chared_id_x)

x_train_expert_features = get_ngramed_id_x(x_train_idx, expert_features_x)
x_test_expert_features = get_ngramed_id_x(x_test_idx, expert_features_x)


###################################### Training #########################################################

def train_dev_step(x, y, emb_mode, is_train=True):
    '''
     if mode_ in [CHAR, CHAR_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_char_id = pad_seq_in_word(x_char_id, FLAGS["data.max_len_chars"])
        x_batch.append(x_char_id)
    if mode_ in [WORD, CHAR_AND_WORD, CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_word_id = pad_seq_in_word(x_word_id, FLAGS["data.max_len_words"])
        x_batch.append(x_word_id)
    if mode_ in [CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_charword_id, x_charword_embedding = pad_seq(x_charword_id, FLAGS["data.max_len_words"],
                                         FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
        x_batch.extend([x_charword_id, x_charword_id_embedding])
    :param x:
    :param y:
    :param emb_mode:
    :param is_train:
    :return:
    '''
    if is_train:
        p = 0.5
    else:
        p = 1.0
    if emb_mode == CHAR:
        feed_dict = {
            cnn.input_expert_feature: x[0],
            cnn.input_x_char_id: x[1],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == WORD:
        feed_dict = {
            cnn.input_expert_feature: x[0],
            cnn.input_x_word_id: x[1],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == CHAR_AND_WORD:
        feed_dict = {
            cnn.input_expert_feature: x[0],
            cnn.input_x_char_id: x[1],
            cnn.input_x_word_id: x[2],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == CHARWORD_AND_WORD:
        feed_dict = {
            cnn.input_expert_feature: x[0],
            cnn.input_x_word_id: x[1],
            cnn.input_x_charword_id: x[2],
            cnn.input_x_charword_id_embedding: x[3],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == CHARWORD_AND_WORD_AND_CHAR:
        feed_dict = {
            cnn.input_expert_feature: x[0],
            cnn.input_x_char_id: x[1],
            cnn.input_x_word_id: x[2],
            cnn.input_x_charword_id: x[3],
            cnn.input_x_charword_id_embedding: x[4],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    if is_train:
        _, step, loss, acc, summary = sess.run([train_op, global_step, cnn.loss, cnn.accuracy,
                                                cnn.merged],
                                            feed_dict)
    else:
        step, loss, acc, summary = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.merged],
                                        feed_dict)
    return step, loss, acc, summary


def make_batches(x_char_id, x_word_id, x_wordchar_id,
                 x_expert_features, y_train, batch_size,
                 nb_epochs, shuffle=False):
    '''
    :param x_char_id:
    :param x_word_id:
    :param x_wordchar_id:
    :param y_train:
    :param batch_size:
    :param nb_epochs:
    :param shuffle:
    :return:
    '''
    mode_ = FLAGS["model.emb_mode"]
    if mode_ == CHAR:
        batch_data = list(zip(x_char_id, x_expert_features, y_train))
    elif mode_ == WORD:
        batch_data = list(zip(x_word_id, x_expert_features, y_train))
    elif mode_ == CHAR_AND_WORD:
        batch_data = list(zip(x_char_id, x_word_id, x_expert_features, y_train))
    elif mode_ == CHARWORD_AND_WORD:
        batch_data = list(zip(x_wordchar_id, x_word_id, x_expert_features,
                              y_train))
    elif mode_ == CHARWORD_AND_WORD_AND_CHAR:
        batch_data = list(zip(x_wordchar_id, x_word_id, x_char_id,
                              x_expert_features, y_train))
    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)

    if nb_epochs > 1:
        nb_batches_per_epoch = int(len(batch_data) / batch_size)
        if len(batch_data) % batch_size != 0:
            nb_batches_per_epoch += 1
        nb_batches = int(nb_batches_per_epoch * nb_epochs)
        return batches, nb_batches_per_epoch, nb_batches
    else:
        return batches


def prep_batches(batch):
    mode_ = FLAGS["model.emb_mode"]
    '''
    x_char: char encoding grouped by word
    x_word: word encoding
    '''
    if mode_ == CHAR:
        x_char_id, x_expert_features, y_batch = zip(*batch)
    elif mode_ == WORD:
        x_word_id, x_expert_features, y_batch = zip(*batch)
    elif mode_ == CHAR_AND_WORD:
        x_char_id, x_word_id, x_expert_features, y_batch = zip(*batch)
    elif mode_ == CHARWORD_AND_WORD:
        x_charword_id, x_word_id, x_expert_features, y_batch = zip(*batch)
    elif mode_ == CHARWORD_AND_WORD_AND_CHAR:
        x_charword_id, x_word_id, x_char_id, x_expert_features, y_batch = zip(*batch)

    x_batch = [np.asarray(x_expert_features)]
    if mode_ in [CHAR, CHAR_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_char_id = pad_seq_in_word(x_char_id, FLAGS["data.max_len_chars"])
        x_batch.append(x_char_id)
    if mode_ in [WORD, CHAR_AND_WORD, CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_word_id = pad_seq_in_word(x_word_id, FLAGS["data.max_len_words"])
        x_batch.append(x_word_id)
    if mode_ in [CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
        x_charword_id, x_charword_id_embedding = pad_seq(x_charword_id, FLAGS["data.max_len_words"],
                                                         FLAGS["data.max_len_subwords"],
                                                         FLAGS["model.emb_dim"])
        x_batch.extend([x_charword_id, x_charword_id_embedding])
    return x_batch, y_batch


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(
            char_ngram_vocab_size=len(ngrams_dict) + 1,
            word_ngram_vocab_size=len(words_dict) + 1,
            char_vocab_size=len(chars_dict) + 1,
            embedding_size=FLAGS["model.emb_dim"],
            word_seq_len=FLAGS["data.max_len_words"],
            char_seq_len=FLAGS["data.max_len_chars"],
            expert_feature_size=expert_features_x.shape[1],
            add_expert_feature=FLAGS["train.add_expert_feature"],
            l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
            mode=FLAGS["model.emb_mode"],
            filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(","))))

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS["train.lr"])
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        print("Writing to {}\n".format(FLAGS["log.output_dir"]))
        if not os.path.exists(FLAGS["log.output_dir"]):
            os.makedirs(FLAGS["log.output_dir"])

        # Save dictionary files
        ngrams_dict_dir = FLAGS["log.output_dir"] + "/subwords_dict.p"
        pickle.dump(ngrams_dict, open(ngrams_dict_dir, "wb"))
        words_dict_dir = FLAGS["log.output_dir"] + "/words_dict.p"
        pickle.dump(words_dict, open(words_dict_dir, "wb"))
        chars_dict_dir = FLAGS["log.output_dir"] + "/chars_dict.p"
        pickle.dump(chars_dict, open(chars_dict_dir, "wb"))

        # Save training and validation logs
        train_log_dir = FLAGS["log.output_dir"] + "/train_logs.csv"
        with open(train_log_dir, "w") as f:
            f.write("step,time,loss,acc\n")
        val_log_dir = FLAGS["log.output_dir"] + "/val_logs.csv"
        with open(val_log_dir, "w")  as f:
            f.write("step,time,loss,acc\n")

        # Save model checkpoints
        checkpoint_dir =  FLAGS["log.output_dir"] + "/checkpoints/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = checkpoint_dir + "model"
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        train_writer = tf.summary.FileWriter(FLAGS["log.output_dir"] + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS["log.output_dir"] + '/test')
        sess.run(tf.global_variables_initializer())

        train_batches, nb_batches_per_epoch, nb_batches = make_batches(x_train_char_id,
                                                                       x_train_word_id,
                                                                       x_train_wordchar_id,
                                                                       x_train_expert_features,
                                                                       y_train_idx,
                                                                       FLAGS["train.batch_size"],
                                                                       FLAGS['train.nb_epochs'],
                                                                       True)

        min_dev_loss = float('Inf')
        dev_loss = float('Inf')
        dev_acc = 0.0
        print("Number of batches in total: {}".format(nb_batches))
        print("Number of batches per epoch: {}".format(nb_batches_per_epoch))

        it = tqdm(range(nb_batches),
                  desc="emb_mode {} delimit_mode {} train_size {}".format(FLAGS["model.emb_mode"],
                                                                          FLAGS[
                                                                              "data.delimit_mode"],
                                                                          x_train_idx.shape[0]),
                  ncols=0)
        for idx in it:
            batch = next(train_batches)
            x_batch, y_batch = prep_batches(batch)
            step, loss, acc, summary = train_dev_step(
                x_batch, y_batch, emb_mode=FLAGS["model.emb_mode"], is_train=True)
            if summary:
                train_writer.add_summary(summary, step)

            if step % FLAGS["log.print_every"] == 0:
                with open(train_log_dir, "a") as f:
                    f.write(
                        "{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(),
                                                       loss, acc))
                it.set_postfix(
                    trn_loss='{:.3e}'.format(loss),
                    trn_acc='{:.3e}'.format(acc),
                    dev_loss='{:.3e}'.format(dev_loss),
                    dev_acc='{:.3e}'.format(dev_acc),
                    min_dev_loss='{:.3e}'.format(min_dev_loss))
            if step % FLAGS["log.eval_every"] == 0 or idx == (nb_batches - 1):
                total_loss = 0
                nb_corrects = 0
                nb_instances = 0
                test_batches = make_batches(x_test_char_id, x_test_word_id, x_test_wordchar_id,
                                            x_test_expert_features,
                                            y_test_idx,
                                            FLAGS['train.batch_size'], 1, False)
                for test_batch in test_batches:
                    x_test_batch, y_test_batch = prep_batches(test_batch)
                    step, batch_dev_loss, batch_dev_acc, summary_dev = train_dev_step(
                        x_test_batch, y_test_batch,
                        emb_mode=FLAGS["model.emb_mode"],
                        is_train=False)
                    nb_instances += x_test_batch[0].shape[0]
                    total_loss += batch_dev_loss * x_test_batch[0].shape[0]
                    nb_corrects += batch_dev_acc * x_test_batch[0].shape[0]
                    if summary:
                        test_writer.add_summary(summary, step)
                dev_loss = total_loss / nb_instances
                dev_acc = nb_corrects / nb_instances

                with open(val_log_dir, "a") as f:
                    f.write(
                        "{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(),
                                                       dev_loss, dev_acc))
                if step % FLAGS["log.checkpoint_every"] == 0 or idx == (nb_batches - 1):
                    if dev_loss < min_dev_loss:
                        path = saver.save(sess, checkpoint_prefix, global_step=step)
                        min_dev_loss = dev_loss
        train_writer.close()
        test_writer.close()
