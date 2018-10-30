from expert_features import get_expert_features
from utils import *
import pickle
import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
import sys

if len(sys.argv) > 2:
    parser = argparse.ArgumentParser(description="Test URLNet model")

    # data args
    default_max_len_words = 200
    parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
                        help="maximum length of url in words (default: {})".format(
                            default_max_len_words))
    default_max_len_chars = 200
    parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
                        help="maximum length of url in characters (default: {})".format(
                            default_max_len_chars))
    default_max_len_subwords = 20
    parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords,
                        metavar="MLSW",
                        help="maxium length of word in subwords/ characters (default: {})".format(
                            default_max_len_subwords))
    parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR",
                        help="location of data file")
    default_delimit_mode = 1
    parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
                        help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(
                            default_delimit_mode))
    parser.add_argument('--data.subword_dict_dir', type=str, default="runs/10000/subwords_dict.p",
                        metavar="SUBWORD_DICT",
                        help="directory of the subword dictionary")
    parser.add_argument('--data.word_dict_dir', type=str, default="runs/10000/word_dict.p",
                        metavar="WORD_DICT",
                        help="directory of the word dictionary")
    parser.add_argument('--data.char_dict_dir', type=str, default="runs/10000/char_dict.p",
                        metavar="	CHAR_DICT",
                        help="directory of the character dictionary")

    # model args
    default_emb_dim = 32
    parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
                        help="embedding dimension size (default: {})".format(default_emb_dim))
    default_emb_mode = 1
    parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
                        help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(
                            default_emb_mode))

    # test args
    default_batch_size = 128
    parser.add_argument('--test.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
                        help="Size of each test batch (default: {})".format(default_batch_size))

    # log args
    parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR",
                        help="directory to save the test results")
    parser.add_argument('--log.checkpoint_dir', type=str, default="runs/10000/checkpoints/",
                        metavar="CHECKPOINTDIR",
                        help="directory of the learned model")

    FLAGS = vars(parser.parse_args())
else:

    CHAR = 1
    WORD = 2
    CHAR_AND_WORD = 3
    CHARWORD_AND_WORD = 4
    CHARWORD_AND_WORD_AND_CHAR = 5

    data_size = 10000
    add_expert_feature = 1
    emb_mode = CHARWORD_AND_WORD_AND_CHAR
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

for key, val in FLAGS.items():
    print("{}={}".format(key, val))

urls, labels = read_data(FLAGS["data.data_dir"])

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)

ngram_dict = pickle.load(open(FLAGS["data.subword_dict_dir"], "rb"))
print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(FLAGS["data.word_dict_dir"], "rb"))
print("size of word vocabulary (train): {}".format(len(word_dict)))
charword_id_x, worded_id_x = ngram_id_x_from_dict(word_x, FLAGS["data.max_len_subwords"], ngram_dict,
                                                  word_dict)
chars_dict = pickle.load(open(FLAGS["data.char_dict_dir"], "rb"))
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])
expert_features_x = np.asarray(get_expert_features(urls))

print("Number of testing urls: {}".format(len(labels)))


######################## EVALUATION ###########################

def test_step(x, emb_mode):
    p = 1.0
    if emb_mode == CHAR:
        feed_dict = {
            input_expert_feature: x[0],
            input_x_char_id: x[1],
            dropout_keep_prob: p
        }
    elif emb_mode == WORD:
        feed_dict = {
            input_expert_feature: x[0],
            input_x_word_id: x[1],
            dropout_keep_prob: p}
    elif emb_mode == CHAR_AND_WORD:
        feed_dict = {
            input_expert_feature: x[0],
            input_x_char_id: x[1],
            input_x_word_id: x[2],
            dropout_keep_prob: p}
    elif emb_mode == CHARWORD_AND_WORD:
        feed_dict = {
            input_expert_feature: x[0],
            input_x_word_id: x[1],
            input_x_charword_id: x[2],
            input_x_charword_id_embedding: x[3],
            dropout_keep_prob: p}
    elif emb_mode == CHARWORD_AND_WORD_AND_CHAR:
        feed_dict = {
            input_expert_feature: x[0],
            input_x_char_id: x[1],
            input_x_word_id: x[2],
            input_x_charword_id: x[3],
            input_x_charword_id_embedding: x[4],
            dropout_keep_prob: p}
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s


checkpoint_file = tf.train.latest_checkpoint(FLAGS["log.checkpoint_dir"])
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_expert_feature = graph.get_operation_by_name('input_expert_feature').outputs[0]

        if FLAGS["model.emb_mode"] in [CHAR, CHAR_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
            input_x_char_id = graph.get_operation_by_name("input_x_char_id").outputs[0]
        if FLAGS["model.emb_mode"] in [CHAR, CHAR_AND_WORD, CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
            input_x_word_id = graph.get_operation_by_name("input_x_word_id").outputs[0]
        if FLAGS["model.emb_mode"] in [CHARWORD_AND_WORD, CHARWORD_AND_WORD_AND_CHAR]:
            input_x_charword_id = graph.get_operation_by_name("input_x_charword_id").outputs[0]
            input_x_charword_id_embedding = graph.get_operation_by_name("input_x_charword_id_embedding").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        if FLAGS["model.emb_mode"] == CHAR:
            batches = batch_iter(list(zip(expert_features_x,chared_id_x)),
                                      FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == WORD:
            batches = batch_iter(list(zip(expert_features_x, worded_id_x)),
                                      FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == CHAR_AND_WORD:
            batches = batch_iter(list(zip(expert_features_x, chared_id_x, worded_id_x)),
                                 FLAGS["test.batch_size"], 1,
                                 shuffle=False)
        elif FLAGS["model.emb_mode"] == CHARWORD_AND_WORD:
            batches = batch_iter(list(zip(expert_features_x, charword_id_x, worded_id_x)),
                                 FLAGS["test.batch_size"], 1,
                                 shuffle=False)
        elif FLAGS["model.emb_mode"] == CHARWORD_AND_WORD_AND_CHAR:
            batches = batch_iter(list(zip(expert_features_x, charword_id_x, worded_id_x,
                                          chared_id_x)),
                                 FLAGS["test.batch_size"], 1, shuffle=False)
        all_predictions = []
        all_scores = []

        nb_batches = int(len(labels) / FLAGS["test.batch_size"])
        if len(labels) % FLAGS["test.batch_size"] != 0:
            nb_batches += 1
        print("Number of batches in total: {}".format(nb_batches))
        it = tqdm(range(nb_batches),
                  desc="emb_mode {} delimit_mode {} test_size {}".format(FLAGS["model.emb_mode"],
                                                                         FLAGS["data.delimit_mode"],
                                                                         len(labels)), ncols=0)
        for idx in it:
            # for batch in batches:
            batch = next(batches)

            if FLAGS["model.emb_mode"] == CHAR:
                x_expert_feature, x_char = zip(*batch)
            elif FLAGS["model.emb_mode"] == WORD:
                x_expert_feature, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == CHAR_AND_WORD:
                x_expert_feature, x_char, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == CHARWORD_AND_WORD:
                x_expert_feature, x_charword, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == CHARWORD_AND_WORD_AND_CHAR:
                x_expert_feature, x_charword, x_word, x_char = zip(*batch)

            x_batch = [x_expert_feature]
            if FLAGS["model.emb_mode"] in [1, 3, 5]:
                x_char = pad_seq_in_word(x_char, FLAGS["data.max_len_chars"])
                x_batch.append(x_char)
            if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
                x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"])
                x_batch.append(x_word)
            if FLAGS["model.emb_mode"] in [4, 5]:
                x_charword, x_char_pad_idx = pad_seq(x_charword, FLAGS["data.max_len_words"],
                                                     FLAGS["data.max_len_subwords"],
                                                     FLAGS["model.emb_dim"])
                x_batch.extend([x_charword, x_char_pad_idx])

            batch_predictions, batch_scores = test_step(x_batch, FLAGS["model.emb_mode"])
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_scores.extend(batch_scores)

            it.set_postfix()

if labels is not None:
    correct_preds = float(sum(all_predictions == labels))
    print("Accuracy: {}".format(correct_preds / float(len(labels))))

save_test_result(labels, all_predictions, all_scores, FLAGS["log.output_dir"])
