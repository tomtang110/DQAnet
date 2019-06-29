from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertLayerNorm, gelu, BertEncoder, BertPooler
import torch
from torch import nn
from utils import generate_data

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob = 0.2, bias = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x

class BertEmbeddings_type(nn.Module):

    def __init__(self,config,max_sentence_type = 10):
        super(BertEmbeddings_type,self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.sentence_type_embeddings = nn.Embedding(max_sentence_type, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, segment_ids,question_type):
        seq_size = token_ids.size()
        position_ids = torch.arange(seq_size[1], dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings((segment_ids > 0).long())
        sentence_type_embeddings = self.sentence_type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings + sentence_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertModel_type(BertModel):

    def __init__(self,config):
        super(BertModel_type,self).__init__(config)
        self.embeddings = BertEmbeddings_type(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
    def forward(self, token_ids, segment_ids=None, attention_mask=None,question_type = None,output_hidden=-4):
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(token_ids, segment_ids,question_type)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        encoded_layers, hidden_layers = encoded_layers[-1], encoded_layers[output_hidden]
        return encoded_layers, hidden_layers


class BertForDQA(BertPreTrainedModel):
    def __init__(self,config):
        super(BertForDQA,self).__init__(config)
        self.bert = BertModel_type(config)
        self.sup_result = nn.Linear(config.hidden_size,4)
        self.apply(self.init_bert_weights)
    def forward(self, args,token_ids,segment_ids,attention_mask,sup_start_labels, sup_end_labels, ans_start_labels, ans_end_labels):
        batch_size = token_ids.size()[0]
        device = token_ids.get_device() if token_ids.is_cuda else torch.device('cpu')
        sequence_output, hidden_output = self.bert(token_ids, segment_ids, attention_mask)

        result_set = self.sup_result(sequence_output)
        sup_start_pro, sup_end_pro, ans_start_pro,ans_end_pro = result_set.split(1,dim=-1)
        sup_start_pro = sup_start_pro.squeeze(-1)
        sup_end_pro = sup_end_pro.squeeze(-1)
        ans_start_pro = ans_start_pro.squeeze(-1)
        ans_end_pro = ans_end_pro.squeeze(-1)
        if sup_start_labels is not None:
            soft_max = torch.nn.LogSoftmax(dim = 1)
            sup_start_loss = -torch.sum(sup_start_labels*soft_max(sup_start_pro))
            sup_end_loss = -torch.sum(sup_end_labels*soft_max(sup_end_pro))
            ans_start_loss = -torch.sum(ans_start_labels*soft_max(ans_start_pro))
            ans_end_loss = -torch.sum(ans_end_labels*soft_max(ans_end_pro))
            sup_loss = torch.mean((sup_start_loss+sup_end_loss))/2
            ans_loss = torch.mean((ans_start_loss+ans_end_loss))/2
            return sup_loss, ans_loss,hidden_output
        else:
            attention_mask = attention_mask.float()
            sup_start_pre = sup_start_pro * attention_mask
            sup_end_pre = sup_end_pro * attention_mask
            for n in range(batch_size):
                sup_start_standard = torch.sum(sup_start_pre[n,0:args.max_s_len])/torch.sum(attention_mask[n,0:args.max_s_len])
                sup_end_standard = torch.sum(sup_end_pre[n,0:args.max_s_len]) / torch.sum(attention_mask[n,0:args.max_s_len])
                sup_standard = sup_start_standard + sup_end_standard
                sup_sentence = []
                for sen in range(1,args.max_s_num+1):
                    if sen != args.max_s_num:
                        sup_start_cur = torch.sum(sup_start_pre[n,args.max_s_len*sen:args.max_s_len*(sen+1)])/torch.sum(attention_mask[n,args.max_s_len*sen:args.max_s_len*(sen+1)])
                        sup_end_cur = torch.sum(
                            sup_end_pre[n, args.max_s_len * sen:args.max_s_len * (sen + 1)]) / torch.sum(
                            attention_mask[n, args.max_s_len * sen:args.max_s_len * (sen + 1)])
                    else:
                        sup_start_cur = torch.sum(
                            sup_start_pre[n, args.max_s_len * sen:]) / torch.sum(
                            attention_mask[n, args.max_s_len * sen:])
                        sup_end_cur = torch.sum(
                            sup_end_pre[n, args.max_s_len * sen:]) / torch.sum(
                            attention_mask[n, args.max_s_len * sen:])
                    sup_cur = sup_start_cur + sup_end_cur
                    if sup_cur > sup_standard:
                        sup_sentence.append((n,sen))
            return sup_sentence,None,hidden_output





class D_attention(nn.Module):
    def __init__(self, input_sizes):
        super(D_attention, self).__init__()
        self.layers = nn.ModuleList()
        head = input_sizes[-1]
        for h in range(head):
            self.layers.append(DQA(input_sizes[0]))
        self.relu = nn.ReLU()
    def forward(self, final_result):
        final_size = final_result.size()
        x = torch.zeros(final_size[-1])
        for i, layer in enumerate(self.layers):
            x += layer(final_result)
        x = self.relu(x/(i+1))
        return x




class DQA(nn.Module):
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)
    def __init__(self,input_size):
        super(DQA, self).__init__()
        self.B_ij = nn.Linear(2*input_size,1)
    def forward(self,attention_mx):
        mx_size = attention_mx.size()
        i_mx = attention_mx[0,:].squeeze(0)
        alph = torch.zeros(mx_size[0])
        for j in range(mx_size[0]):
            ij_cat = (torch.cat([i_mx,attention_mx[j,:]],0)).view(-1)
            B_ij = self.B_ij(ij_cat)
            alph[j] = B_ij
        leakyrelu = nn.LeakyReLU()
        alph = leakyrelu(alph)
        soft_max = nn.Softmax(dim=0)
        pro_j = soft_max(alph).unsqueeze(0)
        final_result = torch.sum(attention_mx * torch.t(pro_j),dim=0)
        return final_result






class DQA_graph(nn.Module):
    def __init__(self,input_size):
        super(DQA_graph,self).__init__()
        self.sentence_level = MLP((input_size[0],input_size[0],1))
        self.dqa = D_attention([input_size[2], input_size[-1]])
        self.yes_no = MLP((input_size[1],input_size[1],1))
        self.pr_node = MLP((input_size[1],input_size[1],2*input_size[1]))
    def forward(self, args,data_set,model,device,pred_mode=False):
        token_ids, segment_ids, input_mask, sup_start_labels, sup_end_labels, \
        ans_start_labels, ans_end_labels, question_type, answer_id, graph = generate_data(args=args,data=data_set,pre=pred_mode)

        sup_loss,ans_loss,hidden_output = model(args,token_ids,segment_ids,input_mask,sup_start_labels,sup_end_labels
                                                ,ans_start_labels,ans_end_labels)

        batch_size = token_ids.size()[0]
        hidden_output = (self.sentence_level(hidden_output)).squeeze(-1)
        own_graph = graph
        for k in range(args.propagate_count):
            new_hidden_output = torch.zeros_like(hidden_output)
            for each_node in own_graph:
                node_num = len(own_graph[each_node])+1
                attention_mx = torch.zeros([node_num,args.max_s_len])
                if each_node[0] == -1:
                    question_vector = torch.zeros([args.max_s_len])
                    q_count = 0
                    for cor_node in own_graph[each_node]:
                        q_count += 1
                        question_vector += hidden_output[cor_node[0],:args.max_s_len]
                    question_vector = question_vector/q_count
                    attention_mx[0,:] = question_vector
                else:
                    attention_mx[0,:] = hidden_output[each_node[0],(each_node[1]+1) * args.max_s_len:(each_node[1] + 2) * args.max_s_len]
                for cor_num, cor_node in enumerate(own_graph[each_node]):
                    if cor_node[1] == -1:
                        attention_mx[cor_num+1,:] = question_vector
                    else:
                        attention_mx[cor_num+1,:] = hidden_output[cor_node[0],(each_node[1]+1) * args.max_s_len:(each_node[1] + 2) * args.max_s_len]

                node_att = self.dqa(attention_mx)
                if each_node[0] != -1:
                    new_hidden_output[each_node[0],(each_node[1]+1) * args.max_s_len:(each_node[1] + 2) * args.max_s_len] = node_att
                    # print(new_hidden_output[each_node[0],:])
            hidden_output = new_hidden_output
        if not pred_mode:
            if question_type == 1:
                ans = torch.tensor([answer_id[0]],dtype = torch.long, device = device)
                ans_yes_no = self.yes_no(hidden_output).squeeze(-1)
                soft_max = torch.nn.Softmax(dim=0)
                ans_yes_no = torch.sum(soft_max(ans_yes_no) * ans_yes_no)
                final_loss = 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(ans_yes_no,ans.to(device))

                return ans_loss,sup_loss,final_loss
            else:
                pred = self.pr_node(hidden_output).view(-1,args.max_p_len,2)
                ans_start_pro, ans_end_pro = pred.split(1, dim=-1)

                ans_start_pro = ans_start_pro.squeeze(-1)
                ans_end_pro = ans_end_pro.squeeze(-1)

                soft_max = torch.nn.LogSoftmax(dim=1)
                ans_start_loss = -torch.sum(ans_start_labels * soft_max(ans_start_pro))
                ans_end_loss = -torch.sum(ans_end_labels * soft_max(ans_end_pro))

                node_loss = torch.mean((ans_start_loss + ans_end_loss)) / 2

                return ans_loss,sup_loss,node_loss
        else:
            if question_type == 1:
                soft_max = torch.nn.Softmax(dim=0)
                ans_yes_no = self.yes_no(hidden_output).squeeze(-1)
                sigmoid = torch.nn.Sigmoid()
                ans_yes_no = sigmoid(torch.sum(soft_max(ans_yes_no) * ans_yes_no))
                return (sup_loss,ans_yes_no.item()>0.5)
            else:
                pred = self.pr_node(hidden_output).view(-1,args.max_p_len,2)
                ans_start_pro, ans_end_pro = pred.split(1, dim=-1)
                ans_start_pro = ans_start_pro.squeeze(-1)
                ans_end_pro = ans_end_pro.squeeze(-1)
                ans_start_pre = ans_start_pro * (input_mask.float())
                ans_end_pre = ans_end_pro * (input_mask.float())
                for n in range(batch_size):
                    ans_start_standard = torch.sum(ans_start_pre[n, 0:args.max_s_len]) / torch.sum(
                        input_mask[n, 0:args.max_s_len])
                    ans_end_standard = torch.sum(ans_end_pre[n, 0:args.max_s_len]) / torch.sum(
                        input_mask[n, 0:args.max_s_len])
                    ans_standard = ans_start_standard + ans_end_standard
                    ans_sentence = []
                    for sen in range(1, args.max_s_num + 1):
                        if sen != args.max_s_num:
                            ans_start_cur = torch.sum(
                                ans_start_pre[n, args.max_s_len * sen:args.max_s_len * (sen + 1)]) / torch.sum(
                                input_mask[n, args.max_s_len * sen:args.max_s_len * (sen + 1)])
                            ans_end_cur = torch.sum(
                                ans_end_pre[n, args.max_s_len * sen:args.max_s_len * (sen + 1)]) / torch.sum(
                                input_mask[n, args.max_s_len * sen:args.max_s_len * (sen + 1)])
                        else:
                            ans_start_cur = torch.sum(
                                ans_start_pre[n, args.max_s_len * sen:]) / torch.sum(
                                input_mask[n, args.max_s_len * sen:])
                            ans_end_cur = torch.sum(
                                ans_end_pre[n, args.max_s_len * sen:]) / torch.sum(
                                input_mask[n, args.max_s_len * sen:])
                        ans_cur = ans_start_cur + ans_end_cur
                        if ans_cur > ans_standard:
                            ans_sentence.append((n, sen,ans_cur))
                if len(ans_sentence) > 1:
                    ans_sentence = max(ans_sentence,key=lambda x:x[0])
                elif len(ans_sentence) == 1:
                    ans_sentence = ans_sentence[0]
                if ans_sentence != []:
                    ans_f_start= ans_start_pre[ans_sentence[0],ans_sentence[1]*args.max_s_len:args.max_s_len * (ans_sentence[1] + 1)]
                    left_a = torch.argmax(ans_f_start.view(-1))
                    ans_f_end = ans_end_pre[ans_sentence[0],ans_sentence[1]*args.max_s_len:args.max_s_len * (ans_sentence[1] + 1)]
                    right_a = torch.argmax(ans_f_end.view(-1))
                    return sup_loss,[ans_sentence[0],ans_sentence[1],left_a.item(),right_a.item()]
                else:
                    return sup_loss,[]







































