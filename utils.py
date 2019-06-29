import re
from fuzzywuzzy import fuzz
from bisect import bisect_left
import numpy as np
import torch
import random
import logging
from collections import Counter
import string
import json
from tqdm import tqdm
GENERAL_WD = ['is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'can', 'could',
              'shall', 'will', 'should', 'would', 'do', 'does', 'did', 'may', 'might', 'must', 'ought', 'need', 'dare']
GENERAL_WD += [x.capitalize() for x in GENERAL_WD]
GENERAL_WD = re.compile(' |'.join(GENERAL_WD))
FIELDS = ['token_ids', 'sup_start_labels', 'sup_end_labels', 'ans_start_labels', 'ans_end_labels', 'segment_ids', 'question_type','answer_id','graph']


class DQA(object):
    pass

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

class WindowMean:
    def __init__(self, window_size = 50):
        self.array = []
        self.sum = 0
        self.window_size = window_size
    def update(self, x):
        self.array.append(x)
        self.sum += x
        if len(self.array) > self.window_size:
            self.sum -= self.array.pop(0)
        return self.sum / len(self.array)

def judge_question_type(q : str, G = GENERAL_WD) -> int:
    if G.match(q):
        return 1
    else:
        return 0

def dp(a, b): # a source, b long text
    f, start = np.zeros((len(a), len(b))), np.zeros((len(a), len(b)), dtype = np.int)
    for j in range(len(b)):
        f[0, j] = int(a[0] != b[j])
        if j > 0 and b[j - 1].isalnum():
            f[0, j] += 10
        start[0, j] = j
    for i in range(1, len(a)):
        for j in range(len(b)):
            # (0, i-1) + del(i) ~ (start[j], j)
            f[i, j] = f[i - 1, j] + 1
            start[i, j] = start[i - 1, j]
            if j == 0:
                continue
            if f[i, j] > f[i - 1, j - 1] + int(a[i] != b[j]):
                f[i, j] = f[i - 1, j - 1] + int(a[i] != b[j])
                start[i, j] = start[i-1, j - 1]

            if f[i, j] > f[i, j - 1] + 0.5:
                f[i, j] = f[i, j - 1] + 0.5
                start[i, j] = start[i, j - 1]
#     print(f[len(a) - 1])
    r = np.argmin(f[len(a) - 1])
    ret = [start[len(a) - 1, r], r + 1]
#     print(b[ret[0]:ret[1]])
    score = f[len(a) - 1, r] / len(a)
    return (ret, score)

def fuzzy_find(entities, sentence):
    ret = []
    for entity in entities:
        item = re.sub(r' \(.*?\)$', '', entity).strip()
        if item == '':
            item = entity
            print(item)
        r, score = dp(item, sentence)
        if score < 0.5:
            matched = sentence[r[0]: r[1]].lower()
            final_word = item.split()[-1]
            # from end
            retry = False
            while fuzz.partial_ratio(final_word.lower(), matched) < 80:
                retry = True
                end = len(item) - len(final_word)
                while end > 0 and item[end - 1].isspace():
                    end -= 1
                if end == 0:
                    retry = False
                    score = 1
                    break
                item = item[:end]
                final_word = item.split()[-1]
            if retry:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                r, score = dp(item, sentence)
                score += 0.1

            if score >= 0.5:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                continue
            del final_word
            # from start
            retry = False
            first_word = item.split()[0]
            while fuzz.partial_ratio(first_word.lower(), matched) < 80:
                retry = True
                start = len(first_word)
                while start < len(item) and item[start].isspace():
                    start += 1
                if start == len(item):
                    retry = False
                    score = 1
                    break
                item = item[start:]
                first_word = item.split()[0]
            if retry:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
                r, score = dp(item, sentence)
                score = max(score, 1 - ((r[1] - r[0]) / len(entity)))
                score += 0.1
#             if score > 0.5:
#                 print(entity + ' ### ' + sentence[r[0]: r[1]] + ' ### ' + item)
            if score < 0.5:
                if item.isdigit() and sentence[r[0]: r[1]] != item:
                    continue
                ret.append((entity, sentence[r[0]: r[1]], int(r[0]), int(r[1]), score))
    non_intersection = []
    for i in range(len(ret)):
        ok = True
        for j in range(len(ret)):
            if j != i:
                if not (ret[i][2] >= ret[j][3] or ret[j][2] >= ret[i][3]) and ret[j][4] < ret[i][4]:
                    ok = False
                    break
                if ret[i][4] > 0.2 and ret[j][4] < 0.1 and not ret[i][1][0].isupper() and len(ret[i][1].split()) <= 3:
                    ok = False
                    print(ret[i])
                    break
        if ok:
            non_intersection.append(ret[i][:4])
    return non_intersection

def find_start_end(tokenizer, tokenized_text, span):
    end_offset, ret = [], []
    for x in tokenized_text:
        offset = len(x) + (end_offset[-1] if len(end_offset) > 0 else -1)
        end_offset.append(offset)
    text = ''.join(tokenized_text)
    t = ''.join(tokenizer.tokenize(span))
    start = text.find(t)
    if start >= 0:
        end = start + len(t) - 1 # include end
    else:
        result = fuzzy_find([t], text)
        if len(result) == 0:
            result = fuzzy_find([re.sub('[UNK]', '',t)], text)
            assert len(result) > 0
        _, _, start, end = result[0]
        end -= 1
    ret.append((bisect_left(end_offset, start), bisect_left(end_offset, end)))
    return ret

class Data_process(object):
    def __init__(self,args,tokenizer,train_files = [],dev_files=[],test_files=[]):
        super(Data_process).__init__()
        self.logger = logging.getLogger('D-QA')
        self.train_set,self.dev_set,self.test_set = [],[],[]
        if train_files:
            self.logger.info('load train dataset....')
            for train_file in train_files:
                with open(train_file,'r') as fin:
                    dataset = json.load(fin)
                count = 0
                for data in tqdm(dataset):
                    try:
                        self.train_set.append(preprocessed_data(args, tokenizer, data))
                    except Exception as error:
                        count += 1
                self.logger.info('There are {} questions excluded'.format(count))
            self.logger.info('Train set size: {} questions'.format(len(self.train_set)))
        if dev_files:
            self.logger.info('load dev dataset....')
            for dev_file in dev_files:
                with open(dev_file,'r') as fin:
                    dataset = json.load(fin)
                count = 0
                for data in tqdm(dataset):
                    try:
                        self.dev_set.append(preprocessed_data(args, tokenizer, data,if_dev=1))
                    except Exception as error:
                        count += 1
                self.logger.info('There are {} questions excluded'.format(count))
            self.logger.info('Dev set size: {} questions'.format(len(self.dev_set)))
        if test_files:
            self.logger.info('load test dataset....')
            for test_file in test_files:
                with open(test_file,'r') as fin:
                    dataset = json.load(fin)
                for data in tqdm(dataset):
                    self.test_set.append(test_preprocessed_data(args, tokenizer, data))
            self.logger.info('Test set size: {} questions'.format(len(self.test_set)))



def test_preprocessed_data(args,tokenizer,data):
    token_ids,segment_ids = [],[]
    token_question = ['[CLS]'] + tokenizer.tokenize(data['question'])
    while len(token_question) < args.max_s_len:
        token_question.append('[PAD]')
    if len(token_question) > args.max_s_len:
        token_question = token_question[:args.max_s_len]
    question_type = judge_question_type(data['question'])
    context = data['context']
    token_id = data['_id']
    total_token = []
    for title_n, para in data['context']:
        token_set = [] + token_question
        segment_id = [0] * len(token_set)
        total_sen = len(para) - 1
        for sen_num, sen in enumerate(para):
            if sen_num+1 > args.max_s_num:
                sen_num -= 1
                break
            token_sentence = ['[SEP]']+tokenizer.tokenize(sen)
            token_len = len(token_sentence)
            while len(token_sentence) < args.max_s_len:
                token_sentence.append('[PAD]')
            if len(token_sentence) > args.max_s_len:
                token_sentence = token_sentence[:args.max_s_len]
            if total_sen == sen_num or sen_num+2 > args.max_s_num:
                token_sentence.append('[SEP]')
            if token_len > len(token_sentence):
                token_len = len(token_sentence)
            token_set += token_sentence
            segment_id += [sen_num + 1] * len(token_sentence)

        while len(token_set) != args.max_p_len:
            segment_id.append(sen_num+2)
            token_set.append('[PAD]')
            assert len(token_set) <= args.max_p_len
        token_ids.append(tokenizer.convert_tokens_to_ids(token_set))
        segment_ids.append(segment_id)
        total_token.append(token_set)
    graph = dict()
    for each_node in data['graph']:
        each_key = eval(each_node)
        if each_key[1] >= args.max_s_num:
            continue
        each_items = data['graph'][each_node]
        each_new_items = []
        for each_item in each_items:
            if each_item[1] >= args.max_s_num:
                continue
            each_new_items.append(each_item)
        graph[each_key] = each_new_items

    one_sample = DQA()
    test_filed = [FIELDS[0],FIELDS[5],FIELDS[6],FIELDS[8],FIELDS[9]]
    for field in test_filed:
        setattr(one_sample, field, eval(field))
    setattr(one_sample, 'total_token', eval('total_token'))
    setattr(one_sample, 'context', eval('context'))
    setattr(one_sample, 'token_id', eval('token_id'))
    return one_sample



def preprocessed_data(args,tokenizer,data,if_dev = 0):
    token_ids, sup_start_labels,sup_end_labels,ans_start_labels,ans_end_labels,segment_ids = \
    [],[],[],[],[],[]
    sup_fact = data['supporting_facts']
    token_question = ['[CLS]'] + tokenizer.tokenize(data['question'])
    while len(token_question) < args.max_s_len:
        token_question.append('[PAD]')
    if len(token_question) > args.max_s_len:
        token_question = token_question[:args.max_s_len]
    question_type = judge_question_type(data['question'])

    if data['answer'] == 'yes' or data['answer'] == 'no':
        question_type = 1
    answer_id = []
    if question_type == 1:
        answer_id.append(int(data['answer'] == 'yes'))
    context = data['context']
    t_id = data['_id']
    total_token = []
    for title_n, para in data['context']:
        token_set = [] + token_question
        segment_id = [0] * len(token_set)
        sup_start_label = [0] * len(token_set)
        sup_end_label = [0] * len(token_set)
        ans_start_label = [0] * len(token_set)
        ans_end_label = [0] * len(token_set)
        sup_start_label[0] = 0.2
        sup_end_label[0] = 0.2
        ans_start_label[0] = 0.2
        ans_end_label[0] = 0.2

        total_sen = len(para) - 1
        for sen_num, sen in enumerate(para):
            if sen_num+1 > args.max_s_num:
                sen_num -= 1
                break
            token_sentence = ['[SEP]']+tokenizer.tokenize(sen)
            token_len = len(token_sentence)
            while len(token_sentence) < args.max_s_len:
                token_sentence.append('[PAD]')
            if len(token_sentence) > args.max_s_len:
                token_sentence = token_sentence[:args.max_s_len]
            if total_sen == sen_num or sen_num+2 > args.max_s_num:
                token_sentence.append('[SEP]')
            if token_len > len(token_sentence):
                token_len = len(token_sentence)
            token_set += token_sentence
            segment_id += [sen_num + 1] * len(token_sentence)
            ss_label = [0] * len(token_sentence)
            se_label = [0] * len(token_sentence)
            as_label = [0] * len(token_sentence)
            ae_label = [0] * len(token_sentence)
            if [title_n,sen_num] in sup_fact:
                if question_type == 0 and data['answer'] in sen:
                    interval = find_start_end(tokenizer,token_sentence,data['answer'])
                    if interval != 0:
                        for (st,en) in interval:
                            as_label[st] = ae_label[en] = 1
                ss_label[1] = se_label[token_len-1] = 1
            sup_start_label += ss_label
            sup_end_label += se_label
            ans_start_label += as_label
            ans_end_label += ae_label

        while len(token_set) != args.max_p_len:
            segment_id.append(sen_num+2)
            token_set.append('[PAD]')
            sup_start_label.append(0)
            sup_end_label.append(0)
            ans_start_label.append(0)
            ans_end_label.append(0)
            assert len(token_set) <= args.max_p_len
        token_ids.append(tokenizer.convert_tokens_to_ids(token_set))
        segment_ids.append(segment_id)
        sup_start_labels.append(sup_start_label)
        sup_end_labels.append(sup_end_label)
        ans_start_labels.append(ans_start_label)
        ans_end_labels.append(ans_end_label)
        if if_dev:
            total_token.append(token_set)

    graph = dict()

    for each_node in data['graph']:
        each_key = eval(each_node)
        if each_key[1] >= args.max_s_num:
            continue
        each_items = data['graph'][each_node]
        each_new_items = []
        for each_item in each_items:
            if each_item[1] >= args.max_s_num:
                continue
            each_new_items.append(each_item)
        graph[each_key] = each_new_items
    one_sample = DQA()
    for field in FIELDS:
        setattr(one_sample, field, eval(field))
    if if_dev:
        setattr(one_sample,'total_token',eval('total_token'))
        setattr(one_sample,'context',eval('context'))
        setattr(one_sample,'t_id',eval('t_id'))
    return one_sample

def generate_data(args,data,l=None,r=None,pre=False):
    if l is None:
        l, r = 0, len(data.token_ids)
    num_in = r - l
    length = len(data.token_ids[0])
    token_id = torch.zeros((num_in,length),dtype = torch.long)
    sup_start_labels = torch.zeros((num_in,length))
    sup_end_labels = torch.zeros((num_in,length))
    ans_start_labels = torch.zeros((num_in,length))
    ans_end_labels = torch.zeros((num_in,length))
    segment_ids = torch.zeros((num_in,length),dtype = torch.long)
    input_mask = torch.zeros((num_in,length),dtype = torch.long)
    for i in range(l,r):
        token_id[i-l,:] = torch.tensor(data.token_ids[i],dtype=torch.long)
        sup_start_labels[i-l,:] = torch.tensor(data.sup_start_labels[i])
        sup_end_labels[i-l,:] = torch.tensor(data.sup_end_labels[i])
        ans_start_labels[i-l,:] = torch.tensor(data.ans_start_labels[i])
        ans_end_labels[i-l,:] = torch.tensor(data.ans_start_labels[i])
        segment_ids[i-l,:] = torch.tensor(data.segment_ids[i])
        input_mask[i-l,:] = (token_id[i-l,:] > 0).long()
    if pre:
        question_type = data.question_type
        graph = data.graph
        return token_id,segment_ids,input_mask,None,None,None,None,question_type,None,graph
    if args.mode == 'fine-tune':
        return token_id, segment_ids, input_mask,sup_start_labels, sup_end_labels, ans_start_labels, ans_end_labels
    else:
        question_type = data.question_type
        answer_id = data.answer_id
        graph = data.graph
        return token_id, segment_ids, input_mask,sup_start_labels, sup_end_labels, ans_start_labels, ans_end_labels,question_type,answer_id,graph


def data_generator(args,dataset):
    if args.mode != 'fine-tune':
        random.shuffle(dataset)
        def generator():
            for data in dataset:
                yield data
        return len(dataset), generator()
    else:
        Bert_set = DQA()
        for field in FIELDS[:6]:
            t = []
            setattr(Bert_set, field, t)
            for data in dataset:
                t.extend(getattr(data, field))
        orders = np.random.permutation(len(t))
        for field in FIELDS[:6]:
            t = getattr(Bert_set, field)
            setattr(Bert_set, field, [t[x] for x in orders])
        num_batch = (len(t) - 1) // args.batch_size + 1
        def generator():
            for batch_num in range(num_batch):
                l, r = batch_num * args.batch_size, min((batch_num + 1) * args.batch_size, len(t))
                yield generate_data(args,Bert_set, l, r)
        return num_batch, generator()

























