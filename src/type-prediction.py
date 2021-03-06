from __future__ import unicode_literals, print_function

import pickle
import sys

import numpy as np
import spacy
from spacy.gold import GoldParse
from spacy.gold import biluo_tags_from_offsets
from spacy.scorer import Scorer

from Embedder import Embedder
from tf_model import TypePredictor, train
from util import inject_tokenizer, read_data, el_hash

max_len = 400


class ClassWeightNormalizer:
    def __init__(self):
        self.class_counter = None

    def init(self, classes):
        import itertools
        from collections import Counter
        self.class_counter = Counter(c for c in itertools.chain.from_iterable(classes))
        total_count = sum(self.class_counter.values())
        self.class_weights = {key: total_count / val for key, val in self.class_counter.items()}

    def __getitem__(self, item):
        return self.class_weights[item]

    def get(self, item, default):
        return self.class_weights.get(item, default)


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return {key: scorer.scores[key] for key in ['ents_p', 'ents_r', 'ents_f', 'ents_per_type']}


def prepare_data(sents):
    sents_w = []
    sents_t = []
    sents_r = []

    nlp = inject_tokenizer(spacy.blank("en"))

    def try_int(val):
        try:
            return int(val)
        except:
            return val

    for s in sents:
        doc = nlp(s[0])
        ents = s[1]['entities']
        repl = s[1]['replacements']

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)

        while "-" in ents_tags:
            ents_tags[ents_tags.index("-")] = "O"

        while "-" in repl_tags:
            repl_tags[repl_tags.index("-")] = "O"

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        sents_w.append(tokens)
        sents_t.append(ents_tags)
        sents_r.append(repl_tags)

    return sents_w, sents_t, sents_r


def load_pkl_emb(path):
    embedder = pickle.load(open(path, "rb"))
    if isinstance(embedder, list):
        embedder = embedder[-1]
    return embedder


def load_w2v_map(w2v_path):

    embs = []
    w_map = dict()

    with open(w2v_path) as w2v:
        n_vectors, n_dims = map(int, w2v.readline().strip().split())
        for ind in range(n_vectors):
            e = w2v.readline().strip().split()

            word = e[0]
            w_map[word] = len(w_map)

            embs.append(list(map(float, e[1:])))

    return Embedder(w_map, np.array(embs))


def create_tag_map(sents):
    tags = set()

    for s in sents:
        tags.update(set(s))

    tagmap = dict(zip(tags, range(len(tags))))

    aid, iid = zip(*tagmap.items())
    inv_tagmap = dict(zip(iid, aid))

    return tagmap, inv_tagmap


def create_batches(batch_size, seq_len, sents, repl, tags, graphmap, wordmap, tagmap, class_weights, element_hash_size=1000):
    pad_id = len(wordmap)
    rpad_id = len(graphmap)
    n_sents = len(sents)

    b_sents = []
    b_repls = []
    b_tags = []
    b_cw = []
    b_lens = []
    b_pref = []
    b_suff = []


    for ind, (s, rr, tt)  in enumerate(zip(sents, repl, tags)):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_r = np.ones((seq_len,), dtype=np.int32) * rpad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)
        blank_cw = np.ones((seq_len,), dtype=np.int32)
        blank_pref = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_suff = np.ones((seq_len,), dtype=np.int32) * element_hash_size


        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_repl = np.array([graphmap.get(r, rpad_id) for r in rr], dtype=np.int32)
        int_tags = np.array([tagmap.get(t, 0) for t in tt], dtype=np.int32)
        int_cw = np.array([class_weights.get(t, 1.0) for t in tt], dtype=np.int32)
        int_pref = np.array([el_hash(w[:3], element_hash_size-1) for w in s], dtype=np.int32)
        int_suff = np.array([el_hash(w[-3:], element_hash_size-1) for w in s], dtype=np.int32)


        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_r[0:min(int_sent.size, seq_len)] = int_repl[0:min(int_sent.size, seq_len)]
        blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]
        blank_cw[0:min(int_sent.size, seq_len)] = int_cw[0:min(int_sent.size, seq_len)]
        blank_pref[0:min(int_sent.size, seq_len)] = int_pref[0:min(int_sent.size, seq_len)]
        blank_suff[0:min(int_sent.size, seq_len)] = int_suff[0:min(int_sent.size, seq_len)]

        b_lens.append(len(s) if len(s) < seq_len else seq_len)
        b_sents.append(blank_s)
        b_repls.append(blank_r)
        b_tags.append(blank_t)
        b_cw.append(blank_cw)
        b_pref.append(blank_pref)
        b_suff.append(blank_suff)

    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    replacements = np.stack(b_repls)
    pos_tags = np.stack(b_tags)
    cw = np.stack(b_cw)
    prefixes = np.stack(b_pref)
    suffixes = np.stack(b_suff)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append({"tok_ids": sentences[i * batch_size: i * batch_size + batch_size, :],
                      "graph_ids": replacements[i * batch_size: i * batch_size + batch_size, :],
                      "prefix": prefixes[i * batch_size: i * batch_size + batch_size, :],
                      "suffix": suffixes[i * batch_size: i * batch_size + batch_size, :],
                      "tags": pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      "class_weights": cw[i * batch_size: i * batch_size + batch_size, :],
                      "lens": lens[i * batch_size: i * batch_size + batch_size]})

    return batch


def parse_biluo(biluo):
    spans = []

    expected = {"B", "U", "0"}
    expected_tag = None

    c_start = 0

    for ind, t in enumerate(biluo):
        if t[0] not in expected:
            expected = {"B", "U", "0"}
            continue

        if t[0] == "U":
            c_start = ind
            c_end = ind + 1
            c_type = t.split("-")[1]
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "B":
            c_start = ind
            expected = {"I", "L"}
            expected_tag = t.split("-")[1]
        elif t[0] == "I":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
        elif t[0] == "L":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
            c_end = ind + 1
            c_type = expected_tag
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "0":
            expected = {"B", "U", "0"}
            expected_tag = None

    return spans


def scorer(pred, labels, inverse_tag_map, eps=1e-8):
    pred_biluo = [inverse_tag_map[p] for p in pred]
    labels_biluo = [inverse_tag_map[p] for p in labels]

    pred_spans = set(parse_biluo(pred_biluo))
    true_spans = set(parse_biluo(labels_biluo))

    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def main_tf(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            output_dir=None, n_iter=30, max_len=100,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0):

    train_s, train_e, train_r = prepare_data(TRAIN_DATA)
    test_s, test_e, test_r = prepare_data(TEST_DATA)

    cw = ClassWeightNormalizer()
    cw.init(train_e)

    t_map, inv_t_map = create_tag_map(train_e)

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    batches = create_batches(32, max_len, train_s, train_r, train_e, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches(len(test_s), max_len, test_s, test_r, test_e, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)

    model = TypePredictor(word_emb, graph_emb, train_embeddings=False,
                 h_sizes=[40, 40, 40], dense_size=30, num_classes=len(t_map),
                 seq_len=max_len, pos_emb_size=30, cnn_win_size=3,
                 suffix_prefix_dims=suffix_prefix_dims, suffix_prefix_buckets=suffix_prefix_buckets)

    train(model=model, train_batches=batches, test_batches=test_batch, epochs=n_iter, learning_rate=learning_rate,
          scorer=lambda pred, true: scorer(pred, true, inv_t_map), learning_rate_decay=learning_rate_decay)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', dest='data_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--graph_emb_path', dest='graph_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--word_emb_path', dest='word_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, type=float,
                        help='')
    parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', default=1.0, type=float,
                        help='')

    args = parser.parse_args()

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = "model-final-ner"
    n_iter = 500

    allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
               'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}


    TRAIN_DATA, TEST_DATA = read_data(args.data_path, normalize=True, allowed=allowed, include_replacements=True)
    main_tf(TRAIN_DATA, TEST_DATA, args.tokenizer,
            graph_emb_path=args.graph_emb_path,
            word_emb_path=args.word_emb_path,
            output_dir=output_dir,
            n_iter=n_iter,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay)