import json
import copy
from tqdm import tqdm
import spacy
nlp_extract = spacy.load('en_core_web_sm')
from fuzzywuzzy import fuzz
from collections import deque

Entity_field = {'CARDINAL','ORDINAL','LANGUAGE','QUANTITY','None'}

def entity_similarity2(s1,s2):
    return fuzz.partial_token_sort_ratio(s1, s2)


def entity_extract(sentence):
    doc = nlp_extract(sentence)
    entities_label = []
    if doc.ents == ():
        entities_label.append((sentence,'None'))
    for ent in doc.ents:
        entities_label.append((ent.text,ent.label_))
    return entities_label

def extract_title_entity(title_enti,context):
    label = set()
    for each in title_enti:
        if each.lower() in context.lower():
            label.add((each.lower(),'TITLE'))
    return label

def delete_para(D_graph):
    abc = deque([(-1,0)])
    sentence_set = set()
    while abc != deque():
        pop_p = abc.popleft()
        if pop_p not in sentence_set:
            sentence_set.add(pop_p)
            new_para = D_graph[pop_p]
        for each in new_para:
            if each not in sentence_set:
                abc.appendleft(each)
    one_set = {each[0] for each in sentence_set}
    k_graph = dict()
    for key_p, obj_p in D_graph.items():
        if key_p[0] in one_set:
            k_graph[key_p] = obj_p
    return k_graph

def add_sup(D_graph,sup_index):
    for each_s in sup_index:
        D_graph[(-1,0)].add(each_s)
        D_graph[each_s].add((-1,0))
        for each_p_s in sup_index:
            if each_p_s != each_s:
                D_graph[each_s].add(each_p_s)
    return D_graph

def to_string(D_graph):
    k_graph = dict()
    for key_t,item_t in D_graph.items():
        k_graph[str(key_t)] = sorted(list(item_t),key=lambda x:(x[0],x[1]))
    return k_graph

def new_dataset(train_set,mode):
    for each_ins in tqdm(train_set):
        D_graph = dict()
        context = each_ins['context']
        question = each_ins['question']
        if mode != 'test':
            sup_fact = each_ins['supporting_facts']
            sup_index = set()
        graph_dict = dict()
        title_entity = set()
        for title, _ in context:
            title_entity.add(title)
        graph_dict['question'] = {each for each in entity_extract(question) if each[1] not in Entity_field}
        graph_dict['question'] |= extract_title_entity(title_entity, question)
        para_count = 0
        for title, para in context:
            sen_dict = dict()
            sen_count = 0
            for sen in para:
                if mode != 'test':
                    for each_sup in sup_fact:
                        if title == each_sup[0] and sen_count == each_sup[1]:
                            sup_index.add((para_count, sen_count))
                sen_dict[sen_count] = {each for each in entity_extract(sen) if each[1] not in Entity_field}
                sen_dict[sen_count] |= extract_title_entity(title_entity, sen)
                #             sen_dict[sen_count].add(title)
                sen_count += 1
            graph_dict[(title, para_count)] = sen_dict
            para_count += 1
        para_key = list(graph_dict.keys())
        cp_para_key = copy.deepcopy(para_key)
        for e_key in para_key:
            if e_key != 'question':
                for e_sen in graph_dict[e_key]:
                    sen_list = set()
                    for oe_sen in graph_dict[e_key]:
                        if e_sen != oe_sen:
                            sen_list.add((e_key[1], oe_sen))
                    for sen_entity in graph_dict[e_key][e_sen]:
                        for other_para in cp_para_key:
                            if other_para == e_key:
                                continue
                            if other_para == 'question':
                                for s_entity in graph_dict[other_para]:
                                    #                                 print(graph_dict[other_para])
                                    if (sen_entity[1] != s_entity[1] or sen_entity[0] == '’s' or s_entity[
                                        0] == '’s') and sen_entity[1] != 'TITLE' and s_entity[1] != 'TITLE':
                                        continue
                                    simi = entity_similarity2(sen_entity[0], s_entity[0])
                                    if simi > 0.8 * 100:
                                        #                                     print(sen_entity[0],'#',s_entity[0],'#',simi)
                                        sen_list.add((-1, 0))
                            else:
                                for other_sentence in graph_dict[other_para]:
                                    for s_entity in graph_dict[other_para][other_sentence]:
                                        if (sen_entity[1] != s_entity[1] or sen_entity[0] == '’s' or s_entity[
                                            0] == '’s') and sen_entity[1] != 'TITLE' and s_entity[1] != 'TITLE':
                                            continue
                                        simi = entity_similarity2(sen_entity[0], s_entity[0])
                                        if simi > 0.8 * 100:
                                            #                                         print(sen_entity[0],'#',s_entity[0],'#',simi)
                                            sen_list.add((other_para[1], other_sentence))
                    D_graph[(e_key[1], e_sen)] = sen_list
            else:
                sen_list = set()
                for e_sen in graph_dict[e_key]:
                    for other_para in cp_para_key:
                        if other_para == e_key:
                            continue
                        for other_sentence in graph_dict[other_para]:
                            for s_entity in graph_dict[other_para][other_sentence]:
                                if (e_sen[1] != s_entity[1] or e_sen[0] == '’s' or s_entity[0] == '’s') and e_sen[
                                    1] != 'TITLE' and s_entity[1] != 'TITLE':
                                    continue
                                simi = entity_similarity2(e_sen[0], s_entity[0])
                                if simi > 0.8 * 100:
                                    sen_list.add((other_para[1], other_sentence))
                #                                 print(e_sen[0],'#',s_entity[0],'#',simi)
                D_graph[(-1, 0)] = sen_list
        if mode != 'test':
            D_graph = add_sup(D_graph, sup_index)
        D_graph = delete_para(D_graph)
        D_graph = to_string(D_graph)
        each_ins['graph'] = D_graph
        # train_graphs.append((D_graph, sup_index))
    with open('./data/hotpot_test_fullwiki_v1_refine.json', 'w') as fout:
        json.dump(train_set, fout)

if __name__ == '__main__':
    with open('./data_undeal/hotpot_test_fullwiki_v1.json', 'r') as fin:
        train_set = json.load(fin)
    print('Finish Reading! len = ', len(train_set))
    # train_small = train_set[:20]
    new_dataset(train_set[:10],'test')
