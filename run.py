import argparse
import logging
import json
import os
from tqdm import tqdm, trange
from hotpot_evaluate_v1 import eval
import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from utils import preprocessed_data,test_preprocessed_data,data_generator,warmup_linear,WindowMean,Data_process
from model import BertForDQA,DQA_graph
from torch.optim import Adam
import joblib as jb
import pickle as plk
import random
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on HOTPOT dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--mode', type=str,
                                default='D-graph',
                                help='fine-tune or D-graph')
    train_settings.add_argument('--Bert_model', type=str,
                               default='bert-base-uncased',
                               help='model of Bert')
    train_settings.add_argument('--at_head', type=int,
                                default=1,
                                help='The head for graph neural network.')
    train_settings.add_argument('--propagate_count',type=int,default=3,
                                help='The number of propagating in graph neural network')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=12,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=1,
                                help='train epochs')
    train_settings.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                help='gradient accumulation steps')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--max_s_num', type=int, default=6,
                                help='max sentence num in one paragraph')
    model_settings.add_argument('--max_p_len', type=int, default=491,
                                help='max length of paragraph')
    model_settings.add_argument('--max_s_len', type=int, default=70,
                                help='max length of sentence')

    path_settings = parser.add_argument_group('path settings')

    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/hotpot_train_v1.1_small_refine.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/hotpot_dev_fullwiki_v1_refine.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/hotpot_test_fullwiki_v1_refine.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='./vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./result/',
                               help='the dir to output the results')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def train_model(args,dataset,model,device,D_model = None):
    logger = logging.getLogger("D-QA")
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    random.shuffle(dataset[0])
    num_batch, dataloader = data_generator(args, dataset[0])
    num_steps = num_batch * args.epochs
    opt = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=num_steps)
    count_step = 0
    if args.mode == 'D-graph':
        opt_dg = Adam(D_model.parameters(), lr=args.learning_rate)
        D_model.to(device)
        D_model.train()
        warmed = False
    model.to(device)
    model.train()
    f1_s_max, em_s_max = -1, -1
    for epoch in trange(args.epochs,desc='Epoch'):
        ans_loss_mean, sup_loss_mean = WindowMean(),WindowMean()
        if args.mode == 'D-graph':
            node_loss_mean = WindowMean()
            opt_dg.zero_grad()
        opt.zero_grad()
        tqdm_obj = tqdm(dataloader, total=num_batch)
        for step, batch in enumerate(tqdm_obj):
            if args.mode == 'fine-tune':
                batch = tuple(t.to(device) for t in batch)
                sup_loss, ans_loss,_ = model(args,*batch)
                loss = ans_loss + sup_loss
            else:
                ans_loss, sup_loss, node_loss = D_model(args,batch, model, device)
                loss = ans_loss + sup_loss + 0.2 * node_loss

            loss.backward()
            if (step+1) % args.gradient_accumulation_steps == 0:
                lr_cur = args.learning_rate * warmup_linear(count_step / num_steps, warmup=0.1)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr_cur
                count_step += 1
                if args.mode == 'D-graph':
                    opt_dg.step()
                    opt_dg.zero_grad()
                    node_mean_loss = node_loss_mean.update(node_loss.item())
                    ans_mean_loss = ans_loss_mean.update(ans_loss.item())
                    sup_mean_loss = sup_loss_mean.update(sup_loss.item())
                    logger.info('ans_loss: {:.2f}, sup_loss: {:.2f}, node_loss: {:.2f}'.format(ans_mean_loss,sup_mean_loss,node_mean_loss))
                    if node_mean_loss < 0.9 and step > 100:
                        warmed = True
                    if warmed:
                        opt.step()
                    opt.zero_grad()
                else:
                    opt.step()
                    opt.zero_grad()
                    ans_mean_loss = ans_loss_mean.update(ans_loss.item())
                    sup_mean_loss = sup_loss_mean.update(sup_loss.item())
                    logger.info('ans_loss: {:.2f}, sup_loss: {:.2f}'.format(ans_mean_loss, sup_mean_loss))
            else:
                if args.mode == 'D-graph':
                    node_loss_mean.update(node_loss.item())
                    ans_loss_mean.update(ans_loss.item())
                    sup_loss_mean.update(sup_loss.item())
                else:
                    ans_loss_mean.update(ans_loss.item())
                    sup_loss_mean.update(sup_loss.item())
            if args.mode == 'fine-tune':
                if step % 1000 == 0:
                    output_model_file = os.path.join(args.model_dir, 'bert-base-uncased.bin.tmp')
                    saved_dict = {'bert-params': model.module.state_dict()}
                    torch.save(saved_dict, output_model_file)
            else:
                if step % 1000 == 0:
                    metircs = evaluate_rd(args,dataset[1],model,D_model,device)
                    if metircs['joint_f1'] > f1_s_max:
                        output_model_file = './models/DQA_model.bin.tmp'
                        saved_dict = {'bert-params': model.module.state_dict()}
                        saved_dict['dg-params'] = D_model.state_dict()
                        torch.save(saved_dict, output_model_file)

    return (model,D_model)





def prepare(args):
    logger = logging.getLogger("D-QA")
    logger.info('Checking the data files...')

    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    tokenizer = BertTokenizer.from_pretrained(args.Bert_model, do_lower_case=True)
    pro_dataset = Data_process(args,tokenizer,train_files=args.train_files,dev_files=args.dev_files)
    train_dev_set = (pro_dataset.train_set,pro_dataset.dev_set)
    with open(os.path.join(args.vocab_dir,'vocab.data'),'wb') as fout:
        plk.dump(train_dev_set,fout)
    logger.info('Done with preparing!')



def train(args):
    logger = logging.getLogger("D-QA")
    # tokenizer = BertTokenizer.from_pretrained(args.Bert_model, do_lower_case=True)
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        dataset = plk.load(fin)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    if args.mode == 'D-graph':
        logger.info('Loading model from {}'.format(os.path.join(args.model_dir,'bert-base-uncased.bin.tmp')))
        model_state_dict = torch.load(os.path.join(args.model_dir,'bert-base-uncased.bin.tmp'))
        model = BertForDQA.from_pretrained(args.Bert_model, state_dict=model_state_dict['bert-params'])
        d_model = DQA_graph((model.config.hidden_size, args.max_p_len, args.max_s_len, args.at_head))
    else:
        model = BertForDQA.from_pretrained(args.Bert_model,cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        d_model = DQA_graph((model.config.hidden_size,args.max_p_len,args.max_s_len,args.at_head))

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    _,_ = train_model(args,dataset,model,device,D_model=d_model)
    logger.info('training is completed!')




def evaluate_rd(args,dev_set,model,d_model,device):
    logger = logging.getLogger("D-QA")
    logger.info('Dev test:')
    num_batch, dataloader = data_generator(args, dev_set)
    tqdm_obj = tqdm(dataloader, total=num_batch)
    sp, answer = {},{}
    for step, batch in enumerate(tqdm_obj):
        original_c = dev_set[step]
        sup_list,ans_list = d_model(args,batch, model, device,pred_mode=True)
        context = original_c.context
        tokens = original_c.total_token
        _id = original_c.t_id
        sup_answer = []
        for each_s in sup_list:
            sup_answer.append((context[each_s[0]][0],each_s[1]))

        if isinstance(ans_list,bool):
            node_answer = ['no','yes'][int(ans_list)]
        elif ans_list == []:
            node_answer = ''
        else:
            node_answer = ' '.join(tokens[ans_list[0]][(ans_list[1])*args.max_s_len:(ans_list[1]+1)*args.max_s_len][ans_list[2]:ans_list[3]+1])

        sp[_id] = sup_answer
        answer[_id] = node_answer
    final_answer = {'answer':answer,'sp':sp}
    with open(os.path.join(args.result_dir,'dev_result.json'),'w') as fout:
        json.dump(final_answer,fout)

    metircs = eval(os.path.join(args.result_dir,'dev_result.json'),args.dev_files[0])
    logger.info('EM: {}, F1: {}, sup_EM: {}, sup_F1: {}, joint_EM: {} joint_F1: {}'.format(
        metircs['em'],metircs['f1'],metircs['sp_em'],metircs['sp_f1'],metircs['joint_em'],metircs['joint_f1']
    ))
    return metircs





def evaluate(args):
    logger = logging.getLogger("D-QA")
    with open(args.dev_files[0] ,'r') as fin:
        dataset = json.load(fin)
    tokenizer = BertTokenizer.from_pretrained(args.Bert_model, do_lower_case=True)
    prodev = []
    for data in dataset:
        prodev.append(preprocessed_data(args,tokenizer,data,if_dev=1))

    logger.info('Loading model ....')
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model_state_dict = torch.load(os.path.join(args.model_dir, 'DQA_model.bin.tmp'))
    model = BertForDQA.from_pretrained(args.Bert_model, state_dict=model_state_dict['bert-params'])
    d_model = DQA_graph((model.config.hidden_size, args.max_p_len, args.max_s_len, args.at_head))
    d_model.load_state_dict(model_state_dict['dg-params'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device).eval()
    d_model.to(device).eval()
    evaluate_rd(args,prodev,model,d_model,device)
    logger.info('result is saved!')


def predict(args):
    logger = logging.getLogger("D-QA")
    with open(args.test_files[0] ,'r') as fin:
        dataset = json.load(fin)
    tokenizer = BertTokenizer.from_pretrained(args.Bert_model, do_lower_case=True)
    protest = []
    for data in dataset:
        protest.append(test_preprocessed_data(args,tokenizer,data))
    logger.info('Loading model ....')
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    model_state_dict = torch.load(os.path.join(args.model_dir, 'DQA_model.bin.tmp'))
    model = BertForDQA.from_pretrained(args.Bert_model, state_dict=model_state_dict['bert-params'])
    d_model = DQA_graph((model.config.hidden_size, args.max_p_len, args.max_s_len, args.at_head))
    d_model.load_state_dict(model_state_dict['dg-params'])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device).eval()
    d_model.to(device).eval()
    logger.info('Dev test:')
    num_batch, dataloader = data_generator(args, protest)
    tqdm_obj = tqdm(dataloader, total=num_batch)
    sp, answer = {}, {}
    for step, batch in enumerate(tqdm_obj):
        original_c = protest[step]
        sup_list, ans_list = d_model(args, batch, model, device, pred_mode=True)
        context = original_c.context
        tokens = original_c.token_set
        _id = original_c.t_id
        sup_answer = []
        for each_s in sup_list:
            sup_answer.append((context[each_s[0]][0], each_s[1]))

        if isinstance(ans_list, bool):
            node_answer = ['no', 'yes'][int(ans_list)]
        else:
            node_answer = ' '.join(tokens[ans_list[0]][ans_list[1]][ans_list[2]:ans_list[3] + 1])

        sp[_id] = sup_answer
        answer[_id] = node_answer
    final_answer = {'answer': answer, 'sp': sp}
    with open(os.path.join(args.result_dir, 'hotpot_test_fullwiki_v1_refine.json_pred'), 'w') as fout:
        json.dump(final_answer, fout)






def run():
    args = parse_args()
    logger = logging.getLogger("D-QA")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))


    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()


