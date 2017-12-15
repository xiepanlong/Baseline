import functools
import numpy
import sklearn.preprocessing

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys
import tensorflow as tf
import toolz
from scipy.sparse import lil_matrix, dok_matrix
from tqdm import tqdm
from utils import citeulike, split_data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


PATH = '/home/xiepanlong/CTRExperiment/Collaborative-metric-learning/Test.txt'

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def save_INFO(path, context, explain):
    """
    :param path: 
    :return: 
    """
    file = open(path, 'a')
    file.write(explain + '\n')
    for i in context:
        file.write(str(i) + ',')
    file.write('\n')
    file.close()

def dcg_at_k(y_score, k=5, method=0):
    y_score = numpy.asfarray(y_score)[:k]
    if y_score.size:
        if method == 0:
            return y_score[0] + numpy.sum(y_score[1:] / numpy.log2(numpy.arange(2, y_score.size + 1)))
        elif method == 1:
            return numpy.sum(y_score / numpy.log2(numpy.arange(2, y_score.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

class RecallEvaluator(object):
    def __init__(self, train_user_item_matrix, test_user_item_matrix):
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix[u, :].nonzero()[1])
                                 for u in range(n_users) if self.test_user_item_matrix[u, :].sum()}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix[u, :].nonzero()[1])
                                      for u in range(n_users) if self.train_user_item_matrix[u, :].sum()}

    def cal_AUC(self, user_id, item_scores):
        """
        Compute AUC
        :param user_id: 
        :param item_scores: 
        :return: 
        """
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        amin, amax = item_scores.min(), item_scores.max()
        item_scores = (item_scores - amin) / (amax - amin)
        idea_score = numpy.zeros(item_scores.shape[0])
        # print('len: ', len(item_scores))
        # print('item_scores: ', item_scores)
        # print('idea_score1: ', idea_score)
        # print('test_set: ')

        for i in test_set:
            # print(i)
            idea_score[i] = 1
        for i in train_set:
            numpy.delete(item_scores, i, axis=0)
            numpy.delete(idea_score, i, axis=0)
        idea_max = idea_score.max()
        # print(idea_max)
        # if idea_max < 1:
        #    return 11111.1
        # print('len2: ', len(idea_score))
        # print('idea_score2: ', idea_score)
        # fpr, tpr, thresholds = roc_curve(idea_score, item_scores, pos_label=2)
        # return auc(fpr, tpr)

        return roc_auc_score(idea_score, item_scores)


    def cal_precision(self, user_id, item_scores, k=50):
        """
        Compute the precision for a particular user given the predicted scores to items
        :param user_id:  the user id 
        :param item_scores: an array contains the predicted score to every item
        :return: precision@K
        """

        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        top_items = numpy.argpartition(-item_scores, k + len(train_set))[:k + len(train_set)]
        top_n_items = 0
        hits = 0
        for i in top_items:
            if i in train_set:
                continue
            if i in test_set:
                hits += 1
            top_n_items += 1
            if top_n_items == k:
                break
        return hits / float(k)

    def cal_NDCG_at_k(self, user_id, item_scores, k=5, method=0):
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        amin, amax = item_scores.min(), item_scores.max()
        y_score = (item_scores - amin) / (amax - amin)

        dcg_max = dcg_at_k(sorted(y_score, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return dcg_at_k(y_score, k, method) / dcg_max

    def cal_Recall(self, user_id, item_scores, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param user_id: the user id
        :param item_scores: an array contains the predicted score to every item
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        top_items = numpy.argpartition(-item_scores, k + len(train_set))[:k + len(train_set)]
        top_n_items = 0
        hits = 0
        for i in top_items:
            if i in train_set:
                continue
            if i in test_set:
                hits += 1
            top_n_items += 1
            if top_n_items == k:
                break
        return hits / float(len(test_set))


class WarpSampler(object):
    """
    A generator that generate tuples: user-positive-item pairs, negative-items

    of shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10):
        self.user_item_matrix = dok_matrix(user_item_matrix)
        self.user_item_pairs = numpy.asarray(self.user_item_matrix.nonzero()).T
        self.batch_size = batch_size
        self.n_negative = n_negative

    @property
    def sample(self):
        while True:
            numpy.random.shuffle(self.user_item_pairs)
            for i in range(int(len(self.user_item_pairs) / self.batch_size)):
                user_positive_items_pairs = self.user_item_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]

                negative_samples = numpy.random.randint(0,
                                                        self.user_item_matrix.shape[1],
                                                        size=(self.batch_size, self.n_negative))

                yield user_positive_items_pairs, negative_samples

    def next_batch(self):
        return self.sample.__next__()


class CML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 features=None,
                 margin=0.1,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.features = None

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor

        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize

    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=self.features,
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        return tf.layers.dense(inputs=dropout, units=self.embed_dim, name="mlp_layer_2")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """

        # feature loss
        if self.features is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor

            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        if self.feature_projection is not None:

            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_mean(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)

            # apply regularization weight
            return tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg
        else:
            return tf.constant(0, dtype=tf.float32)

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        # positive item to user distance (N)
        pos_distances = tf.reduce_mean(tf.squared_difference(users, pos_items), 1, name="pos_distances")

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples),
                                 (0, 2, 1), name="neg_items")
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_mean(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1,
                                               name="distance_to_neg_items")

        # number of impostors for each user-positive-item pair
        impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")

        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_distances + self.margin, 0,
                                   name="pair_loss")
        weighted_loss_per_pair = loss_per_pair

        # the embedding loss
        loss = tf.reduce_sum(weighted_loss_per_pair, name="loss")

        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        return self.embedding_loss + self.feature_loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []
        gds.append(tf.train
                   .AdagradOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))
        if self.feature_projection:
            gds.append(tf.train
                       .AdagradOptimizer(self.master_learning_rate)
                       .minimize(self.feature_loss / self.n_items))

        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        # score = minus distance (N_USER, N_ITEM)
        return -tf.reduce_sum(tf.squared_difference(user, item), 2, name="scores")


BATCH_SIZE = 50000
N_NEGATIVE = 20
EVALUATION_EVERY_N_BATCHES = 100
EMBED_DIM = 50


def optimize(model, sampler, train, valid):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if model.feature_projection is not None:
        # initialize item embedding with feature projection
        sess.run(tf.assign(model.item_embeddings, model.feature_projection))
    while True:
        # create evaluator on validation set
        validation_recall = RecallEvaluator(train, valid)
        # compute recall on validate set
        valid_recalls = []
        # sample some users to calculate recall validation
        valid_users = list(set(valid.nonzero()[0]))[:]
        for user_chunk in toolz.partition_all(len(valid_users), valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_recalls.extend([validation_recall.cal_Recall(user, user_scores)
                                  for user, user_scores in zip(user_chunk, scores)]
                                 )
        print("\nRecall on (sampled) validation set: {}".format(numpy.mean(valid_recalls)))
        # TODO: early stopping based on validation recall

        # compute precision
        valid_precisions = []
        # sample some users to calculate precision validation
        valid_users = list(set(valid.nonzero()[0]))[:]
        for user_chunk in toolz.partition_all(len(valid_users), valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_precisions.extend([validation_recall.cal_precision(user, user_scores)
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )
        print("\nPrecision on (sampled) validation set: {}".format(numpy.mean(valid_precisions)))

        # compute AUC
        vaild_AUCs = []
        # sample some users to calculate precision validation
        valid_users = list(set(valid.nonzero()[0]))[:]
        for user_chunk in toolz.partition_all(len(valid_users), valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            vaild_AUCs.extend([validation_recall.cal_AUC(user, user_scores)
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )
        print("\nAUC on (sampled) validation set: {}".format(numpy.mean(vaild_AUCs)))

        #Compute NDCG
        valid_NDCGs = []
        valid_users = list(set(valid.nonzero()[0]))[:]
        for user_chunk in toolz.partition_all(len(valid_users), valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_NDCGs.extend([validation_recall.cal_NDCG_at_k(user, user_scores, len(valid_users))
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )

        print("\nNDCG on (sampled) validation set: {}".format(numpy.mean(valid_NDCGs)))
        # train model
        losses = []
        # run n mini-batches
        for _ in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Optimizing..."):
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses.append(loss)
        print("\nTraining loss {}".format(numpy.mean(losses)))


if __name__ == '__main__':
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10
    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)
    # create warp sampler
    sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    model = CML(n_users,
                n_items,
                # set features to None to disable feature projection
                features=None,
                # size of embedding
                embed_dim=EMBED_DIM,
                # the size of hinge loss margin.
                margin=1.0,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=1.1,
                # learning rate for AdaGrad
                master_learning_rate=0.9,
                )

    optimize(model, sampler, train, valid)

    # WITH features
    # In this case, we additionally train a feature projector to project raw item features into the
    # embedding. The projection serves as "a prior" to inform the item's potential location in the embedding.
    # We use a two fully-connected layers NN as our feature projector. (This model is much more computation intensive.
    # A GPU machine is recommended)
    model = CML(n_users,
                n_items,
                # enable feature projection
                features=dense_features,
                embed_dim=EMBED_DIM,
                margin=1.0,
                clip_norm=1.5,
                master_learning_rate=0.9,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim=256,
                # dropout rate between hidden layer and output layer in the feature projector NN
                dropout_rate=0.5,
                # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                feature_projection_scaling_factor=1,
                # the penalty to the distance between projection and item's actual location in the embedding
                # tune this to adjust how much the embedding should be biased towards the item features.
                feature_l2_reg=5,
                )
    optimize(model, sampler, train, valid)



PATH = '/home/xiepanlong/CTRExperiment/Collaborative-metric-learning/Test.txt'

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def save_INFO(path, context, explain):
    """
    :param path: 
    :return: 
    """
    file = open(path, 'a')
    file.write(explain + '\n')
    for i in context:
        file.write(str(i) + ',')
    file.write('\n')
    file.close()

def dcg_at_k(y_score, k=5, method=0):
    y_score = numpy.asfarray(y_score)[:k]
    if y_score.size:
        if method == 0:
            return y_score[0] + numpy.sum(y_score[1:] / numpy.log2(numpy.arange(2, y_score.size + 1)))
        elif method == 1:
            return numpy.sum(y_score / numpy.log2(numpy.arange(2, y_score.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

class RecallEvaluator(object):
    def __init__(self, train_user_item_matrix, test_user_item_matrix):
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix[u, :].nonzero()[1])
                                 for u in range(n_users) if self.test_user_item_matrix[u, :].sum()}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix[u, :].nonzero()[1])
                                      for u in range(n_users) if self.train_user_item_matrix[u, :].sum()}

    def cal_AUC(self, user_id, item_scores):
        """
        Compute AUC
        :param user_id: 
        :param item_scores: 
        :return: 
        """
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        amin, amax = item_scores.min(), item_scores.max()
        item_scores = (item_scores - amin) / (amax - amin)
        idea_score = numpy.zeros(item_scores.shape[0])

        for i in test_set:
            idea_score[i] = 1
        for i in train_set:
            numpy.delete(item_scores, i, axis=0)
            numpy.delete(idea_score, i, axis=0)
        idea_max = idea_score.max()
        if idea_max < 1:
            return 11111.1

        return roc_auc_score(idea_score, item_scores)


    def cal_precision(self, user_id, item_scores, k=50):
        """
        Compute the precision for a particular user given the predicted scores to items
        :param user_id:  the user id 
        :param item_scores: an array contains the predicted score to every item
        :return: precision@K
        """

        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        top_items = numpy.argpartition(-item_scores, k + len(train_set))[:k + len(train_set)]
        top_n_items = 0
        hits = 0
        for i in top_items:
            if i in train_set:
                continue
            if i in test_set:
                hits += 1
            top_n_items += 1
            if top_n_items == k:
                break
        return hits / float(k)

    def cal_Recall(self, user_id, item_scores, k=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param user_id: the user id
        :param item_scores: an array contains the predicted score to every item
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        top_items = numpy.argpartition(-item_scores, k + len(train_set))[:k + len(train_set)]
        top_n_items = 0
        hits = 0
        for i in top_items:
            if i in train_set:
                continue
            if i in test_set:
                hits += 1
            top_n_items += 1
            if top_n_items == k:
                break
        return hits / float(len(test_set))

     def cal_NDCG_at_k(self, user_id, item_scores, k=5, method=0):
        train_set = self.user_to_train_set[user_id]
        test_set = self.user_to_test_set[user_id]

        amin, amax = item_scores.min(), item_scores.max()
        y_score = (item_scores - amin) / (amax - amin)

        dcg_max = dcg_at_k(sorted(y_score, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return dcg_at_k(y_score, k, method) / dcg_max

class WarpSampler(object):
    """
    A generator that generate tuples: user-positive-item pairs, negative-items

    of shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, batch_size=10000, n_negative=10):
        self.user_item_matrix = dok_matrix(user_item_matrix)
        self.user_item_pairs = numpy.asarray(self.user_item_matrix.nonzero()).T
        self.batch_size = batch_size
        self.n_negative = n_negative

    @property
    def sample(self):
        while True:
            numpy.random.shuffle(self.user_item_pairs)
            for i in range(int(len(self.user_item_pairs) / self.batch_size)):
                user_positive_items_pairs = self.user_item_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]

                negative_samples = numpy.random.randint(0,
                                                        self.user_item_matrix.shape[1],
                                                        size=(self.batch_size, self.n_negative))

                yield user_positive_items_pairs, negative_samples

    def next_batch(self):
        return self.sample.__next__()


class CML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 features=None,
                 margin=0.1,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.features = None

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor

        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize

    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=self.features,
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        return tf.layers.dense(inputs=dropout, units=self.embed_dim, name="mlp_layer_2")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """

        # feature loss
        if self.features is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor

            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        if self.feature_projection is not None:

            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_mean(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)

            # apply regularization weight
            return tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg
        else:
            return tf.constant(0, dtype=tf.float32)

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        # positive item to user distance (N)
        pos_distances = tf.reduce_mean(tf.squared_difference(users, pos_items), 1, name="pos_distances")

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples),
                                 (0, 2, 1), name="neg_items")
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_mean(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1,
                                               name="distance_to_neg_items")

        # number of impostors for each user-positive-item pair
        impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0

        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")

        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_distances + self.margin, 0,
                                   name="pair_loss")
        weighted_loss_per_pair = loss_per_pair

        # the embedding loss
        loss = tf.reduce_sum(weighted_loss_per_pair, name="loss")

        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        return self.embedding_loss + self.feature_loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []
        gds.append(tf.train
                   .AdagradOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))
        if self.feature_projection:
            gds.append(tf.train
                       .AdagradOptimizer(self.master_learning_rate)
                       .minimize(self.feature_loss / self.n_items))

        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        # score = minus distance (N_USER, N_ITEM)
        return -tf.reduce_sum(tf.squared_difference(user, item), 2, name="scores")


BATCH_SIZE = 50000
N_NEGATIVE = 20
EVALUATION_EVERY_N_BATCHES = 100
EMBED_DIM = 50


def optimize(model, sampler, train, valid):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if model.feature_projection is not None:
        # initialize item embedding with feature projection
        sess.run(tf.assign(model.item_embeddings, model.feature_projection))
    while True:
        # create evaluator on validation set
        validation_recall = RecallEvaluator(train, valid)
        # compute recall on validate set
        valid_recalls = []
        # sample some users to calculate recall validation
        valid_users = list(set(valid.nonzero()[0]))[:300]
        for user_chunk in toolz.partition_all(300, valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_recalls.extend([validation_recall.cal_Recall(user, user_scores)
                                  for user, user_scores in zip(user_chunk, scores)]
                                 )
        print("\nRecall on (sampled) validation set: {}".format(numpy.mean(valid_recalls)))
        # TODO: early stopping based on validation recall

        # compute precision
        valid_precisions = []
        # sample some users to calculate precision validation
        valid_users = list(set(valid.nonzero()[0]))[:300]
        for user_chunk in toolz.partition_all(300, valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_precisions.extend([validation_recall.cal_precision(user, user_scores)
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )
        print("\nPrecision on (sampled) validation set: {}".format(numpy.mean(valid_precisions)))

        # compute AUC
        vaild_AUCs = []
        # sample some users to calculate precision validation
        valid_users = list(set(valid.nonzero()[0]))[:300]
        for user_chunk in toolz.partition_all(300, valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            vaild_AUCs.extend([validation_recall.cal_AUC(user, user_scores)
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )
        print("\nAUC on (sampled) validation set: {}".format(numpy.mean(vaild_AUCs)))

        #Compute NDCG
        valid_NDCGs = []
        valid_users = list(set(valid.nonzero()[0]))[:]
        for user_chunk in toolz.partition_all(len(valid_users), valid_users):
            scores = sess.run(model.item_scores, {model.score_user_ids: user_chunk})
            valid_NDCGs.extend([validation_recall.cal_NDCG_at_k(user, user_scores, len(valid_users))
                                     for user, user_scores in zip(user_chunk, scores)]
                                    )

        print("\nNDCG on (sampled) validation set: {}".format(numpy.mean(valid_NDCGs)))

        # train model
        losses = []
        # run n mini-batches
        for _ in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Optimizing..."):
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses.append(loss)
        print("\nTraining loss {}".format(numpy.mean(losses)))


if __name__ == '__main__':
    # get user-item matrix
    user_item_matrix, features = citeulike(tag_occurence_thres=5)
    n_users, n_items = user_item_matrix.shape
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10
    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)
    # create warp sampler
    sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
    model = CML(n_users,
                n_items,
                # set features to None to disable feature projection
                features=None,
                # size of embedding
                embed_dim=EMBED_DIM,
                # the size of hinge loss margin.
                margin=1.0,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=1.1,
                # learning rate for AdaGrad
                master_learning_rate=0.9,
                )

    optimize(model, sampler, train, valid)

    # WITH features
    # In this case, we additionally train a feature projector to project raw item features into the
    # embedding. The projection serves as "a prior" to inform the item's potential location in the embedding.
    # We use a two fully-connected layers NN as our feature projector. (This model is much more computation intensive.
    # A GPU machine is recommended)
    model = CML(n_users,
                n_items,
                # enable feature projection
                features=dense_features,
                embed_dim=EMBED_DIM,
                margin=1.0,
                clip_norm=1.5,
                master_learning_rate=0.9,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim=256,
                # dropout rate between hidden layer and output layer in the feature projector NN
                dropout_rate=0.5,
                # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                feature_projection_scaling_factor=1,
                # the penalty to the distance between projection and item's actual location in the embedding
                # tune this to adjust how much the embedding should be biased towards the item features.
                feature_l2_reg=5,
                )
    optimize(model, sampler, train, valid)
