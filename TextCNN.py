import tensorflow as tf

CHAR = 1
WORD = 2
CHAR_AND_WORD = 3
CHARWORD_AND_WORD = 4
CHARWORD_AND_WORD_AND_CHAR = 5

NUM_FILTER = 256


class TextCNN(object):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, \
                 word_seq_len, char_seq_len, embedding_size, expert_feature_size,
                 add_expert_feature,
                 l2_reg_lambda=0, \
                 filter_sizes=[3, 4, 5, 6], mode=0):
        '''
        1: only character CNN,
        2: only word CNN,
        3: character and word CNN,
        4: character-level-word and word CNN,
        5: character-level-word and word and character CNN

        '''
        #charword
        if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
            self.input_x_charword_id = tf.placeholder(tf.int32, [None, None, None],
                                                      name="input_x_charword_id")
            self.input_x_charword_id_embedding = tf.placeholder(tf.float32,
                                                                [None, None, None, embedding_size],
                                                                name="input_x_charword_id_embedding")
        #word
        if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR or mode == WORD or mode == CHAR_AND_WORD:
            self.input_x_word_id = tf.placeholder(tf.int32, [None, None], name="input_x_word_id")

        #char
        if mode == CHAR or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
            self.input_x_char_id = tf.placeholder(tf.int32, [None, None], name="input_x_char_id")

        self.input_expert_features = tf.placeholder(tf.float32, [None, expert_feature_size],
                                                        name="input_expert_feature")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            # charword
            if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.charword_embedding = tf.Variable(
                    tf.random_uniform([char_ngram_vocab_size, embedding_size], -1.0, 1.0),
                    name="charword_embedding")

            # word
            if mode == WORD or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.word_embedding = tf.Variable(
                    tf.random_uniform([word_ngram_vocab_size, embedding_size], -1.0, 1.0),
                    name="word_embedding")

            # char
            if mode == CHAR or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.char_embedding = tf.Variable(
                    tf.random_uniform([char_vocab_size, embedding_size], -1.0, 1.0),
                    name="char_embedding")

                # charword
            if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.embedded_x_charword = tf.nn.embedding_lookup(self.charword_embedding, self.input_x_charword_id)
                self.embedded_x_charword = tf.multiply(self.embedded_x_charword,
                                                       self.input_x_charword_id_embedding)

            if mode == WORD or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.embedded_x_word = tf.nn.embedding_lookup(self.word_embedding, self.input_x_word_id)

            if mode == CHAR or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.embedded_x_char = tf.nn.embedding_lookup(self.char_embedding,
                                                              self.input_x_char_id)

            if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.sum_ngram_x_char = tf.reduce_sum(self.embedded_x_charword, 2)
                self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)

            # expand dimension to fit to conv2d model
            if mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
            if mode == WORD or mode == CHAR_AND_WORD:
                self.sum_ngram_x_expanded = tf.expand_dims(self.embedded_x_word, -1)
            if mode == CHAR or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
                self.char_x_expanded = tf.expand_dims(self.embedded_x_char, -1)

                ########################### WORD CONVOLUTION LAYER ################################
        if mode == WORD or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
            pooled_x = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, NUM_FILTER]
                    b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER]), name="b")
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.sum_ngram_x_expanded,
                        w,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, word_seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_x.append(pooled)

            num_filters_total = NUM_FILTER * len(filter_sizes)
            self.h_pool = tf.concat(pooled_x, 3)
            self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")
            self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x")

            ########################### CHAR CONVOLUTION LAYER ###########################
        if mode == CHAR or mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
            pooled_char_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, NUM_FILTER]
                    b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER]), name="b")
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.char_x_expanded,
                        w,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, char_seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_char_x.append(pooled)
            num_filters_total = NUM_FILTER * len(filter_sizes)
            self.h_char_pool = tf.concat(pooled_char_x, 3)
            self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total],
                                          name="pooled_char_x")
            self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob,
                                             name="dropout_char_x")

        ############################### CONCAT WORD AND CHAR BRANCH ############################
        if mode == CHAR_AND_WORD or mode == CHARWORD_AND_WORD_AND_CHAR:
            with tf.name_scope("word_char_concat"):
                weight_word = tf.get_variable("ww", shape=(num_filters_total, 512),
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias_word = tf.Variable(tf.constant(0.1, shape=[512]), name="bw")
                l2_loss += tf.nn.l2_loss(weight_word)
                l2_loss += tf.nn.l2_loss(bias_word)
                word_output = tf.nn.xw_plus_b(self.h_drop, weight_word, bias_word)

                weight_char = tf.get_variable("wc", shape=(num_filters_total, 512),
                                     initializer=tf.contrib.layers.xavier_initializer())
                bias_char = tf.Variable(tf.constant(0.1, shape=[512]), name="bc")
                l2_loss += tf.nn.l2_loss(weight_char)
                l2_loss += tf.nn.l2_loss(bias_char)
                char_output = tf.nn.xw_plus_b(self.char_h_drop, weight_char, bias_char)

                self.conv_output = tf.concat([word_output, char_output], 1)
        elif mode == WORD or mode == CHARWORD_AND_WORD:
            self.conv_output = self.h_drop
        elif mode == CHAR:
            self.conv_output = self.char_h_drop

        ############################### HUMAN Expert Features ############################
        if add_expert_feature:
            self.conv_output = tf.concat([self.conv_output, self.input_expert_features], 1)
        ################################ RELU AND FC ###################################
        with tf.name_scope("output"):
            input_dim = 1024
            if add_expert_feature:
                input_dim += expert_feature_size
            w0 = tf.get_variable("w0", shape=[input_dim, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b0")
            l2_loss += tf.nn.l2_loss(w0)
            l2_loss += tf.nn.l2_loss(b0)
            output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)

            w1 = tf.get_variable("w1", shape=[512, 256],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1")
            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)
            output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)

            w2 = tf.get_variable("w2", shape=[256, 128],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
            l2_loss += tf.nn.l2_loss(w2)
            l2_loss += tf.nn.l2_loss(b2)
            output2 = tf.nn.relu(tf.matmul(output1, w2) + b2)

            w = tf.get_variable("w", shape=(128, 2),
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(output2, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                             labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()